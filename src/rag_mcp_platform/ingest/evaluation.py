"""Offline extraction-quality evaluation against a user-supplied JSON manifest."""

import json
from pathlib import Path

from .indexer import inspect_sources


def _contains(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _contains(actual[key], value) for key, value in expected.items())
    return actual == expected


def evaluate_manifest(path: str) -> dict:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("Evaluation manifest requires version 1.")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation manifest requires a nonempty cases list.")
    results = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not isinstance(case.get("path"), str):
            raise ValueError(f"Evaluation case {index} requires a source path.")
        for field in ("expected_text", "forbidden_text"):
            if field in case and (not isinstance(case[field], list) or not all(isinstance(value, str) for value in case[field])):
                raise ValueError(f"Evaluation case {index}: {field} must be a list of strings.")
        if "expected_metadata" in case and not isinstance(case["expected_metadata"], dict):
            raise ValueError(f"Evaluation case {index}: expected_metadata must be an object.")
        failures = []
        try:
            source = manifest_path.parent / case["path"]
            report = inspect_sources(str(source), preview=True)
            if report["files_seen"] != 1:
                raise ValueError("Each evaluation case must select exactly one supported source file.")
            result = report["results"][0]
            if result["status"] != case.get("expected_status", "ready"):
                failures.append("status")
            chunks = result.get("chunks", [])
            text = "\n\n".join(
                "\n".join([*chunk.get("metadata", {}).get("section_path", []), chunk.get("heading") or "", chunk["content"]])
                for chunk in chunks
            )
            for position, fragment in enumerate(case.get("expected_text", [])):
                if fragment not in text:
                    failures.append(f"expected_text[{position}]")
            for position, fragment in enumerate(case.get("forbidden_text", [])):
                if fragment in text:
                    failures.append(f"forbidden_text[{position}]")
            if not _contains(result.get("metadata", {}), case.get("expected_metadata", {})):
                failures.append("metadata")
            for position, section_path in enumerate(case.get("expected_section_paths", [])):
                if not any(chunk["metadata"].get("section_path") == section_path for chunk in chunks):
                    failures.append(f"expected_section_paths[{position}]")
            if "max_chunk_tokens" in case and any(chunk["token_count"] > case["max_chunk_tokens"] for chunk in chunks):
                failures.append("max_chunk_tokens")
            if "min_chunks" in case and len(chunks) < case["min_chunks"]:
                failures.append("min_chunks")
            if "expected_error_contains" in case and case["expected_error_contains"] not in result.get("error", ""):
                failures.append("error_category")
            results.append({"case": index, "passed": not failures, "failed_checks": failures, "chunk_count": len(chunks)})
        except Exception:
            results.append({"case": index, "passed": False, "failed_checks": ["inspection_error"]})
    passed = sum(result["passed"] for result in results)
    return {"cases": len(results), "passed": passed, "failed": len(results) - passed,
            "status": "success" if passed == len(results) else "failed", "results": results}