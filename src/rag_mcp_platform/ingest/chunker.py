"""Heading-aware, token-windowed chunking with overlap."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from markdown_it import MarkdownIt


@lru_cache
def _markdown_parser():
    return MarkdownIt("commonmark").enable("table")


@dataclass
class Chunk:
    index: int
    heading: str | None
    content: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@lru_cache
def _token_len():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except Exception:  # pragma: no cover - fallback when tiktoken unavailable
        return lambda text: max(1, len(text.split()))


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    return [(heading, body) for heading, body, _ in _section_records(text)]


def _section_records(text: str) -> list[tuple[str | None, str, list[str]]]:
    """Split top-level headings without interpreting code as document structure."""
    sections: list[tuple[str | None, str, list[str]]] = []
    hierarchy: list[tuple[int, str]] = []
    current_heading: str | None = None
    lines = text.splitlines(keepends=True)
    tokens = _markdown_parser().parse(text)
    start = 0
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.level != 0 or token.map is None:
            continue
        body = "".join(lines[start:token.map[0]]).strip("\r\n")
        if body.strip():
            sections.append((current_heading, body, [heading for _, heading in hierarchy]))
        current_heading = tokens[index + 1].content.strip()
        level = int(token.tag[1:])
        while hierarchy and hierarchy[-1][0] >= level:
            hierarchy.pop()
        hierarchy.append((level, current_heading))
        start = token.map[1]
    body = "".join(lines[start:]).strip("\r\n")
    if body.strip():
        sections.append((current_heading, body, [heading for _, heading in hierarchy]))
    return sections


def _markdown_blocks(body: str) -> list[tuple[str, bool]]:
    lines = body.splitlines(keepends=True)
    protected = {"fence", "code_block", "table_open", "bullet_list_open", "ordered_list_open"}
    return [
        ("".join(lines[token.map[0]:token.map[1]]).rstrip("\r\n"), token.type in protected)
        for token in _markdown_parser().parse(body)
        if token.map is not None and token.level == 0 and token.nesting != -1
    ]


def _split_prose(text: str, max_tokens: int) -> list[str]:
    token_len = _token_len()
    pieces = []
    remaining = text
    while remaining and token_len(remaining) > max_tokens:
        lower, upper = 1, len(remaining)
        length = 0
        while lower <= upper:
            middle = (lower + upper) // 2
            if token_len(remaining[:middle]) <= max_tokens:
                length = middle
                lower = middle + 1
            else:
                upper = middle - 1
        if not length:
            raise ValueError("CHUNK_MAX_TOKENS is too small to encode a source character.")
        boundary = remaining.rfind(" ", 0, length + 1)
        if boundary > length // 2:
            length = boundary
        piece = remaining[:length].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[length:].lstrip()
    if remaining.strip():
        pieces.append(remaining.strip())
    return pieces


def _split_structured(text: str, max_tokens: int) -> list[str]:
    token_len = _token_len()
    tokens = _markdown_parser().parse(text)
    token = tokens[0]
    lines = text.splitlines(keepends=True)
    prefix, suffix = "", ""
    if token.type == "table_open":
        prefix = "".join(lines[:2]).rstrip("\r\n") + "\n"
        units = lines[2:]
    elif token.type == "fence":
        prefix = token.markup + token.info + "\n"
        suffix = "\n" + token.markup
        units = token.content.splitlines(keepends=True)
    elif token.type in {"bullet_list_open", "ordered_list_open"}:
        units = [
            "".join(lines[item.map[0]:item.map[1]])
            for item in tokens if item.type == "list_item_open" and item.level == 1 and item.map
        ]
    else:
        units = [text]

    def render(parts):
        return prefix + "".join(parts).rstrip("\r\n") + suffix

    pieces = []
    current = []
    for unit in units:
        if token_len(render([unit])) > max_tokens:
            raise ValueError(
                "Structured block exceeds CHUNK_MAX_TOKENS: a single row, code line, "
                "list item, or its repeated header cannot fit without losing structure."
            )
        if current and token_len(render([*current, unit])) > max_tokens:
            pieces.append(render(current))
            current = []
        current.append(unit)
    if current:
        pieces.append(render(current))
    if not pieces:
        raise ValueError("Structured block exceeds CHUNK_MAX_TOKENS and has no splittable units.")
    return pieces


def _window(body: str, max_tokens: int, overlap_tokens: int) -> list[tuple[str, int]]:
    token_len = _token_len()
    paragraphs = []
    for block, protected in _markdown_blocks(body):
        if token_len(block) <= max_tokens:
            paragraphs.append(block)
        elif protected:
            paragraphs.extend(_split_structured(block, max_tokens))
        else:
            paragraphs.extend(_split_prose(block, max_tokens))

    windows: list[tuple[str, int]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            chunk_text = "\n\n".join(current)
            windows.append((chunk_text, token_len(chunk_text)))
            current = []

    for para in paragraphs:
        if token_len("\n\n".join([*current, para])) > max_tokens and current:
            tail = current[-1]
            flush()
            if (
                0 < token_len(tail) <= overlap_tokens
                and token_len(f"{tail}\n\n{para}") <= max_tokens
            ):
                current = [tail]

        current.append(para)

    flush()
    return windows


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Chunk limits require max_tokens > 0 and 0 <= overlap_tokens < max_tokens.")
    chunks: list[Chunk] = []
    index = 0
    for heading, body, section_path in _section_records(text):
        for content, tokens in _window(body, max_tokens, overlap_tokens):
            chunks.append(Chunk(
                index=index, heading=heading, content=content, token_count=tokens,
                metadata={"section_path": section_path} if section_path else {},
            ))
            index += 1
    return chunks
