"""Model-free Presidio detection and replacement without logging detected values."""

import json
import re
import hashlib
from collections import Counter
from dataclasses import replace
from functools import lru_cache

from .loaders import LoadedDoc, LoadedPage


class PiiRejectedError(ValueError):
    """Sensitive data could not be processed safely, or a strict audit found PII."""


_PERSON_NAME_PATTERN = re.compile(
    r"(?i:\b(?:full\s+name|contact\s+name|contact\s+person|customer|patient|"
    r"employee|recipient|submitted\s+by|prepared\s+by|attention)\s*[:=]\s*"
    r")([A-Z][A-Za-z'’-]{1,39}(?:\s+[A-Z][A-Za-z'’-]{1,39}){1,3})\b"
)
_STREET_SUFFIX = (
    r"street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|"
    r"drive|dr\.?|court|ct\.?|circle|cir\.?|highway|hwy\.?|parkway|pkwy\.?|"
    r"place|pl\.?|terrace|ter\.?|trail|trl\.?|way"
)
_LABELLED_STREET_PATTERN = re.compile(
    r"(?i:\b(?:home\s+address|mailing\s+address|street\s+address|address)\s*[:=]\s*)"
    r"[^\n;]+"
)
_POSTAL_ADDRESS_PATTERN = re.compile(
    rf"\b\d{{1,6}}\s+(?:[\w.'’-]+\s+){{0,6}}(?:{_STREET_SUFFIX})"
    rf"\s*,\s*[A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+)*"
    rf"(?:\s*,?\s*[A-Z]{{2}})?\s+\d{{5}}(?:-\d{{4}})?\b",
    re.IGNORECASE,
)
@lru_cache
def _get_recognizers():
    from presidio_analyzer import Pattern, PatternRecognizer
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer, EmailRecognizer, PhoneRecognizer, UsSsnRecognizer,
    )

    return (
        EmailRecognizer(), PhoneRecognizer(supported_regions=["US"]),
        CreditCardRecognizer(), UsSsnRecognizer(),
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS", supported_language="en",
            patterns=[Pattern(
                "email_shaped_source_text",
                r"\b[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+\b", 0.85,
            )],
            global_regex_flags=re.MULTILINE | re.IGNORECASE,
        ),
        PatternRecognizer(
            supported_entity="PERSON", supported_language="en",
            patterns=[Pattern("labelled_english_name", _PERSON_NAME_PATTERN.pattern, 0.85)],
            global_regex_flags=re.MULTILINE,
        ),
        PatternRecognizer(
            supported_entity="POSTAL_ADDRESS", supported_language="en",
            patterns=[
                Pattern("labelled_address", _LABELLED_STREET_PATTERN.pattern, 0.85),
                Pattern("full_postal_address", _POSTAL_ADDRESS_PATTERN.pattern, 0.85),
            ],
            global_regex_flags=re.MULTILINE | re.IGNORECASE,
        ),
        PatternRecognizer(
            supported_entity="US_SSN", supported_language="en",
            patterns=[Pattern(
                "separated_ssn", r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", 0.85,
            )],
        ),
    )


def _presidio_results(text: str):
    from presidio_analyzer import RecognizerResult

    findings = []
    for recognizer in _get_recognizers():
        for result in recognizer.analyze(text=text, entities=recognizer.supported_entities):
            threshold = 0.4 if result.entity_type == "PHONE_NUMBER" else 0.5
            if result.score < threshold:
                continue
            start, end = result.start, result.end
            if result.entity_type == "PERSON":
                match = _PERSON_NAME_PATTERN.search(text, start, end)
                if match:
                    start, end = match.span(1)
            elif result.entity_type == "POSTAL_ADDRESS":
                label = re.match(r"(?i:(?:home\s+address|mailing\s+address|street\s+address|address)\s*[:=]\s*)", text[start:end])
                if label:
                    start += label.end()
                if text[start:end].strip() == "<POSTAL_ADDRESS>":
                    continue
            if start < end:
                findings.append(RecognizerResult(result.entity_type, start, end, result.score))
    return findings


def _presidio_counts(text: str) -> Counter:
    categories = {
        "EMAIL_ADDRESS": "email", "PHONE_NUMBER": "phone", "CREDIT_CARD": "credit_card",
        "US_SSN": "ssn", "PERSON": "person_name", "POSTAL_ADDRESS": "postal_address",
    }
    spans: dict[str, list[tuple[int, int]]] = {}
    for result in _presidio_results(text):
        spans.setdefault(categories[result.entity_type], []).append((result.start, result.end))
    counts = Counter()
    for category, findings in spans.items():
        merged = []
        for start, end in sorted(findings):
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        counts[category] = len(merged)
    return counts


@lru_cache
def _get_anonymizer():
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


def _replace_text(text: str, counts: Counter, *, field_name: str = "") -> str:
    from presidio_anonymizer.entities import OperatorConfig

    context = ""
    key = field_name.lower().replace("_", " ")
    if key in {"author", "full name", "contact name", "contact person", "customer", "patient", "employee", "recipient"}:
        context = "Contact Name: "
    elif key in {"address", "home address", "mailing address", "street address"}:
        context = "Address: "
    source = context + text
    results = _presidio_results(source)
    if not results:
        return text
    operators = {result.entity_type: OperatorConfig("replace", {"new_value": f"<{result.entity_type}>"}) for result in results}
    sanitized = _get_anonymizer().anonymize(text=source, analyzer_results=results, operators=operators)
    counts.update(item.entity_type for item in sanitized.items)
    if _presidio_results(sanitized.text):
        raise ValueError("Sanitization did not remove all detected identifiers.")
    if context and not sanitized.text.startswith(context):
        raise ValueError("Sanitization changed field context unexpectedly.")
    return sanitized.text[len(context):]


def _replace_metadata(value, counts: Counter, *, field_name=""):
    if isinstance(value, dict):
        sanitized = {}
        for key, child in value.items():
            safe_key = _replace_text(str(key), counts)
            if safe_key in sanitized:
                raise ValueError("Sanitized metadata keys collide.")
            sanitized[safe_key] = _replace_metadata(child, counts, field_name=str(key))
        return sanitized
    if isinstance(value, list):
        return [_replace_metadata(child, counts, field_name=field_name) for child in value]
    if isinstance(value, str):
        return _replace_text(value, counts, field_name=field_name)
    if type(value) in {int, float}:
        text = _replace_text(str(value), counts, field_name=field_name)
        return text if text != str(value) else value
    return value


def sanitize_document(document: LoadedDoc) -> LoadedDoc:
    """Replace detected identifiers before hashing, chunking, storage, or preview."""
    try:
        counts = Counter()
        text = _replace_text(document.text, counts)
        title = _replace_text(document.title, counts)
        metadata = _replace_metadata(document.metadata, counts)
        pages = [LoadedPage(
            page.page_number, _replace_text(page.text, counts),
            _replace_text(page.raw_text, counts) if page.raw_text is not None else None,
        ) for page in document.pages]
        identity = _replace_text(document.source_path, counts)
        if identity != document.source_path:
            identity = "sanitized://" + hashlib.sha256(document.source_path.encode("utf-8")).hexdigest()
        url = document.url
        if url and _replace_text(url, counts) != url:
            url = None
        if counts:
            metadata["pii_sanitization"] = {"policy": "replace", "replacements": dict(counts)}
        return replace(document, text=text, title=title, metadata=metadata, pages=pages, source_path=identity, url=url)
    except Exception:
        raise PiiRejectedError("PII sanitization unavailable or incomplete; document was not indexed.") from None


def reject_pii(document: LoadedDoc) -> None:
    """Reject supported structured identifiers in text, metadata, or source labels."""
    text = "\n".join([
        document.text, document.title, document.source_path, document.url or "",
        json.dumps(document.metadata, ensure_ascii=False),
        *[page.text for page in document.pages if page.text not in document.text],
        *[page.raw_text for page in document.pages if page.raw_text and page.raw_text != page.text],
    ])
    try:
        counts = _presidio_counts(text)
    except Exception:
        raise PiiRejectedError("PII screening unavailable; document rejected without indexing.") from None
    if counts:
        summary = ", ".join(f"{category}={counts[category]}" for category in sorted(counts))
        raise PiiRejectedError(f"PII policy rejected document ({summary}). No content was indexed.")