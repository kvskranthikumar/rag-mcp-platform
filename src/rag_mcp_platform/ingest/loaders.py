"""Format-specific loaders with structured text and source-page provenance."""

from __future__ import annotations

from collections.abc import Iterator
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..settings import get_settings

SUPPORTED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
}


@dataclass
class LoadedPage:
    page_number: int
    text: str
    raw_text: str | None = None


@dataclass
class LoadedDoc:
    source_path: str
    doc_type: str
    title: str
    text: str
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[LoadedPage] = field(default_factory=list)


def iter_files(root: str | Path) -> Iterator[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Documentation path does not exist: {root_path}")
    if root_path.is_file():
        if root_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root_path
        else:
            raise ValueError(f"Unsupported documentation file type: {root_path.suffix}")
        return
    for path in sorted(root_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _pdf_table_text(page, tables, excluded_lines: set[tuple[int, int]]) -> str:
    import pymupdf

    regions = [pymupdf.Rect(table.bbox) for table in tables]
    regions.extend(pymupdf.Rect(table.header.bbox) for table in tables if table.header.external)
    events = [(table.bbox[1], table.bbox[0], table.to_markdown()) for table in tables]
    lines: dict[tuple[int, int], list] = {}
    for word in page.get_text("words", sort=True):
        key = (word[5], word[6])
        center = pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
        if key not in excluded_lines and not any(center in region for region in regions):
            lines.setdefault(key, []).append(word)
    for words in lines.values():
        events.append((min(word[1] for word in words), min(word[0] for word in words),
                       " ".join(word[4] for word in words)))
    return "\n\n".join(text for _, _, text in sorted(events))


def _pdf_repeated_margins(document) -> set[tuple[str, str]]:
    occurrences: Counter = Counter()
    for page in document:
        candidates = set()
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = " ".join("".join(span["text"] for span in line["spans"]).split())
                if not text:
                    continue
                if line["bbox"][3] <= page.rect.height * 0.1:
                    candidates.add(("top", text))
                elif line["bbox"][1] >= page.rect.height * 0.9:
                    candidates.add(("bottom", text))
        occurrences.update(candidates)
    return {key for key, count in occurrences.items() if count >= max(3, (document.page_count * 3 + 4) // 5)}


def _load_pdf(path: Path) -> tuple[list[LoadedPage], dict[str, Any]]:
    import pymupdf

    suppress_recommendation = getattr(pymupdf, "no_recommend_layout", None)
    if suppress_recommendation is not None:
        suppress_recommendation()
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDF must be decrypted before ingestion.")
        settings = get_settings()
        repeated = _pdf_repeated_margins(document) if settings.pdf_remove_repeated_margins else set()
        pages = []
        table_count = removed_lines = 0
        for index, page in enumerate(document, 1):
            raw_text = page.get_text("text", sort=True)
            excluded_lines = set()
            if repeated:
                for block_index, block in enumerate(page.get_text("dict")["blocks"]):
                    for line_index, line in enumerate(block.get("lines", [])):
                        text = " ".join("".join(span["text"] for span in line["spans"]).split())
                        position = "top" if line["bbox"][3] <= page.rect.height * 0.1 else "bottom" if line["bbox"][1] >= page.rect.height * 0.9 else None
                        if (position, text) in repeated:
                            excluded_lines.add((block.get("number", block_index), line_index))
            finder = page.find_tables() if settings.pdf_extract_tables else None
            tables = finder.tables if finder is not None else []
            text = _pdf_table_text(page, tables, excluded_lines) if tables or excluded_lines else raw_text
            pages.append(LoadedPage(index, text, raw_text))
            table_count += len(tables)
            removed_lines += len(excluded_lines)
        metadata = {
            "page_count": document.page_count,
            "pages_without_text": [page.page_number for page in pages if not page.text.strip()],
            "extraction_method": "pymupdf_text",
            "pdf_metadata": document.metadata or {},
            "table_count": table_count,
            "removed_repeated_margin_lines": removed_lines,
        }
    if not any(page.text.strip() for page in pages):
        raise ValueError(
            "PDF contains no extractable text. OCR or a text-enabled PDF is required "
            "before ingestion."
        )
    return pages, metadata


def _to_markdown(content) -> str:
    from markdownify import markdownify

    text = markdownify(
        str(content), heading_style="ATX", strip_pre=None,
        table_infer_header=False, wrap=False,
    )
    return text.strip("\r\n")


def _docx_html_blocks(container, soup, *, in_cell: bool = False):
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            if not block.text.strip():
                continue
            style = (block.style.name or "").lower() if block.style else ""
            tag = "p"
            if not in_cell and style.startswith("heading"):
                level = "".join(character for character in style if character.isdigit()) or "1"
                tag = f"h{max(1, min(int(level), 6))}"
            elif not in_cell and style == "title":
                tag = "h1"
            paragraph = soup.new_tag(tag)
            paragraph.string = block.text
            yield paragraph
        elif isinstance(block, Table):
            if in_cell:
                for row_index, row in enumerate(block.rows, 1):
                    for column_index, cell in enumerate(row.cells, row.grid_cols_before + 1):
                        children = list(_docx_html_blocks(cell, soup, in_cell=True))
                        if not children:
                            continue
                        entry = soup.new_tag("div")
                        label = soup.new_tag("p")
                        label.string = f"Nested table row {row_index}, column {column_index}:"
                        entry.append(label)
                        for child in children:
                            entry.append(child)
                        yield entry
                continue
            table = soup.new_tag("table")
            for row in block.rows:
                rendered_row = soup.new_tag("tr")
                for _ in range(row.grid_cols_before):
                    rendered_row.append(soup.new_tag("td"))
                for cell in row.cells:
                    rendered_cell = soup.new_tag("td")
                    for child in _docx_html_blocks(cell, soup, in_cell=True):
                        rendered_cell.append(child)
                    rendered_row.append(rendered_cell)
                for _ in range(row.grid_cols_after):
                    rendered_row.append(soup.new_tag("td"))
                table.append(rendered_row)
            if table.get_text(strip=True):
                yield table


def _load_docx(path: Path) -> str:
    import docx
    from bs4 import BeautifulSoup

    document = docx.Document(str(path))
    soup = BeautifulSoup("", "lxml")
    content = soup.new_tag("div")
    for block in _docx_html_blocks(document, soup):
        content.append(block)
    return _to_markdown(content)


def _normalize_html_code(soup, content) -> None:
    """Group the consecutive code-line paragraphs used by MadCap automation help."""
    from bs4 import NavigableString, Tag

    for paragraph in list(content.select("p.courierline")):
        if paragraph.parent is None:
            continue
        members = []
        sibling = paragraph
        while sibling is not None:
            if isinstance(sibling, NavigableString) and not sibling.strip():
                sibling = sibling.next_sibling
                continue
            if not isinstance(sibling, Tag) or sibling.name != "p":
                break
            if "courierline" not in sibling.get("class", []):
                break
            members.append(sibling)
            sibling = sibling.next_sibling
        pre = soup.new_tag("pre")
        code = soup.new_tag("code")
        code.string = "\n".join(member.get_text().replace("\xa0", " ") for member in members)
        pre.append(code)
        paragraph.insert_before(pre)
        for member in members:
            member.extract()


def _html_metadata(soup) -> tuple[dict[str, Any], str | None]:
    fields = {
        "product": "product", "version": "product_version", "author": "author",
        "category": "category", "document": "document_type", "audience": "audience",
    }
    metadata: dict[str, Any] = {}
    for tag in soup.select("meta[name][content]"):
        name = tag["name"].strip().lower()
        value = tag["content"].strip()
        if name in fields and value:
            metadata[fields[name]] = value
        elif name == "keywords" and value:
            metadata["tags"] = list(dict.fromkeys(
                keyword.strip() for keyword in value.split(",") if keyword.strip()
            ))
    if soup.html:
        language = soup.html.get("lang") or soup.html.get("xml:lang")
        if language and language.strip():
            metadata["language"] = language.strip()
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    if title:
        metadata["source_title"] = title
    canonical = soup.find("link", rel="canonical")
    url = None
    if canonical and canonical.get("href"):
        candidate = canonical["href"].strip()
        try:
            parsed = urlsplit(candidate)
            if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
                url = candidate
        except ValueError:
            pass
    return metadata, url


def _load_html(path: Path) -> tuple[str, str | None, dict[str, Any], str | None]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    metadata, url = _html_metadata(soup)
    content = (
        soup.find(id="mc-main-content")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    for tag in content.select(
        "head, script, style, nav, footer, button, form, noscript, template, "
        "[role='navigation' i], .nocontent, .topicToolbarProxy, .breadcrumb, .breadcrumbs"
    ):
        tag.decompose()
    _normalize_html_code(soup, content)
    return _to_markdown(content), title or None, metadata, url


def _derive_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return path.stem


def load_file(path: Path) -> LoadedDoc | None:
    ext = path.suffix.lower()
    html_title: str | None = None
    pages: list[LoadedPage] = []
    metadata: dict[str, Any] = {}
    url: str | None = None

    if ext in {".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_type = "markdown"
    elif ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_type = "text"
    elif ext == ".pdf":
        pages, metadata = _load_pdf(path)
        text = "\n\n".join(page.text for page in pages)
        doc_type = "pdf"
    elif ext == ".docx":
        text = _load_docx(path)
        doc_type = "docx"
    elif ext in {".html", ".htm"}:
        text, html_title, metadata, url = _load_html(path)
        doc_type = "html"
    else:
        return None

    if not text.strip():
        return None

    pdf_title = (metadata.get("pdf_metadata", {}).get("title") or "").strip()
    title = html_title or pdf_title or _derive_title(path, text)
    return LoadedDoc(
        source_path=str(path),
        doc_type=doc_type,
        title=title,
        text=text,
        metadata=metadata,
        pages=pages,
        url=url,
    )
