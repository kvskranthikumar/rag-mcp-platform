"""Incremental ingestion: hash-based change detection, embed, upsert."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from pathlib import Path

import numpy as np
from psycopg.rows import dict_row

from ..db import get_pool
from ..providers.embeddings import get_embedding_provider
from ..settings import get_settings
from ..telemetry import (
    capture_metrics, create_trace_id, increment_metric, measured, observation, stage,
)
from .chunker import Chunk, chunk_text
from .loaders import LoadedDoc, iter_files, load_file
from .pii import PiiRejectedError, sanitize_document


_EMBEDDING_SPACE_LOCK = (1296257107, 1)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document_content_hash(document: LoadedDoc) -> str:
    metadata = {
        key: value for key, value in document.metadata.items()
        if key not in {"processing_fingerprint", "ingestion_pipeline_version", "original_filename"}
    }
    if not document.pages and not metadata and not document.url:
        return _content_hash(document.text)
    content = (
        [{"page_number": page.page_number, "text": page.text} for page in document.pages]
        if document.pages else document.text
    )
    return _content_hash(json.dumps({
        "content": content, "metadata": metadata, "url": document.url,
    }, sort_keys=True, separators=(",", ":")))


def _validate_chunks(chunks: list[Chunk], max_tokens: int) -> None:
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk.content, str) or not chunk.content.strip():
            raise ValueError(f"Chunk {index} has empty or invalid content.")
        if type(chunk.index) is not int or chunk.index != index:
            raise ValueError(f"Chunk {index} must have sequential index {index}.")
        if type(chunk.token_count) is not int or not 0 < chunk.token_count <= max_tokens:
            raise ValueError(
                f"Chunk {index} token count must be between 1 and {max_tokens}; "
                f"got {chunk.token_count!r}."
            )


def _validate_embeddings(vectors, chunk_count: int, dimension: int) -> np.ndarray:
    try:
        count = len(vectors)
    except TypeError as exc:
        raise ValueError("Embedding response must contain one vector per chunk.") from exc
    if count != chunk_count:
        raise ValueError(f"Expected {chunk_count} embedding vectors, received {count}.")
    try:
        values = np.asarray(vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding vectors must be rectangular numeric arrays.") from exc
    if values.shape != (chunk_count, dimension):
        raise ValueError(
            f"Expected embedding shape ({chunk_count}, {dimension}), got {values.shape}."
        )
    if values.dtype.kind not in "iuf":
        raise ValueError("Embedding vectors must contain real numeric values.")
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Embedding vectors must contain finite float32 values.")
    if not np.any(values != 0, axis=1).all():
        raise ValueError("Embedding vectors must be nonzero for cosine search.")
    return values


def _require_compatible_embedding_space(conn, signature: str, dimension: int, *, lock=False):
    if lock:
        conn.execute("SELECT pg_advisory_xact_lock(%s, %s)", _EMBEDDING_SPACE_LOCK)
    column = conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding' AND NOT attisdropped"
    ).fetchone()
    expected_type = f"vector({dimension})"
    if column is None or column[0] != expected_type:
        actual_type = column[0] if column else "missing"
        raise ValueError(
            f"Embedding column is {actual_type}, but configuration requires {expected_type}. "
            "Build a complete index in a separate database with the configured dimension."
        )
    incompatible = conn.execute(
        "SELECT embedding_signature FROM documents "
        "WHERE embedding_signature IS DISTINCT FROM %s LIMIT 1",
        (signature,),
    ).fetchone()
    if incompatible is not None:
        raise ValueError(
            f"Index contains a different or unknown embedding signature "
            f"({incompatible[0]!r}); configured signature is {signature!r}. "
            "In-place embedding-space changes are blocked, including --full. "
            "Build a complete index in a separate database, verify it, then switch "
            "all ingestion and search clients together."
        )


def _read_document(cur, source_path: str, *, lock: bool = False):
    query = (
        "SELECT id, content_hash, embedding_signature, metadata, version "
        "FROM documents WHERE source_path = %s"
    )
    if lock:
        query += " FOR UPDATE"
    cur.execute(query, (source_path,))
    return cur.fetchone()


def _skip_reason(cur, existing, digest: str, signature: str, fingerprint: str) -> str | None:
    if existing:
        if (
            existing["content_hash"] == digest
            and existing["embedding_signature"] == signature
            and (existing["metadata"] or {}).get("processing_fingerprint") == fingerprint
        ):
            return "skipped"
        return None
    cur.execute(
        "SELECT source_path FROM documents "
        "WHERE content_hash = %s AND embedding_signature = %s "
        "AND metadata->>'processing_fingerprint' = %s LIMIT 1",
        (digest, signature, fingerprint),
    )
    return "duplicate" if cur.fetchone() else None


def _prepare_chunks(doc: LoadedDoc, settings) -> list[Chunk]:
    if doc.pages:
        chunks = []
        for page in doc.pages:
            for chunk in chunk_text(page.text, settings.chunk_max_tokens, settings.chunk_overlap_tokens):
                chunk.index = len(chunks)
                chunk.metadata["page_number"] = page.page_number
                chunks.append(chunk)
    else:
        chunks = chunk_text(doc.text, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    shared_metadata = {
        key: doc.metadata[key] for key in (
            "product", "product_version", "language", "author", "category", "document_type",
            "audience", "tags", "source_title", "original_filename",
            "processing_fingerprint", "ingestion_pipeline_version", "pii_sanitization",
        ) if key in doc.metadata
    }
    shared_metadata.update(source_path=doc.source_path, doc_type=doc.doc_type)
    for chunk in chunks:
        chunk.metadata = {**shared_metadata, **chunk.metadata}
    _validate_chunks(chunks, settings.chunk_max_tokens)
    return chunks


def _hide_rejected_identity(result: dict) -> None:
    identity = result["source_path"]
    result["source_path"] = "rejected://" + _content_hash(identity)
    if "file" in result:
        result["file"] = "[PII rejected]"


def _safe_result_identity(result: dict) -> None:
    try:
        document = sanitize_document(LoadedDoc(
            result["source_path"], "text", result.get("file", ""), "",
        ))
        if document.source_path != result["source_path"]:
            result["source_path"] = document.source_path
            if "file" in result:
                result["file"] = "[sanitized source]"
        elif "file" in result:
            result["file"] = document.title
    except PiiRejectedError:
        _hide_rejected_identity(result)


def inspect_sources(path: str | None = None, *, preview: bool = False) -> dict:
    """Extract, chunk, and report quality without database or provider calls."""
    settings = get_settings()
    results = []
    seen_hashes: dict[str, str] = {}
    for file_path in iter_files(path or settings.docs_source_dir):
        result = {"source_path": str(file_path), "status": "empty", "warnings": []}
        try:
            document = load_file(file_path)
            if document is not None:
                document = sanitize_document(document)
                result["source_path"] = document.source_path
                document.metadata.update(
                    processing_fingerprint=settings.processing_fingerprint,
                    ingestion_pipeline_version=settings.ingestion_pipeline_version,
                )
                chunks = _prepare_chunks(document, settings)
                digest = _document_content_hash(document)
                if digest in seen_hashes:
                    result["warnings"].append(f"Duplicate content of {seen_hashes[digest]}")
                seen_hashes[digest] = document.source_path
                if document.doc_type == "html":
                    missing = [key for key in ("product", "product_version", "language") if key not in document.metadata]
                    if missing:
                        result["warnings"].append("Missing source metadata: " + ", ".join(missing))
                if document.metadata.get("pages_without_text"):
                    result["warnings"].append("Pages without text: " + str(document.metadata["pages_without_text"]))
                result.update(
                    status="ready" if chunks else "empty", title=document.title,
                    metadata=document.metadata, chunk_count=len(chunks),
                    total_tokens=sum(chunk.token_count for chunk in chunks),
                    max_chunk_tokens=max((chunk.token_count for chunk in chunks), default=0),
                )
                if preview:
                    result["chunks"] = [
                        {"index": chunk.index, "heading": chunk.heading, "content": chunk.content,
                         "token_count": chunk.token_count, "metadata": chunk.metadata}
                        for chunk in chunks
                    ]
        except Exception as exc:
            result.update(status="failed", error=str(exc) or type(exc).__name__)
            if isinstance(exc, PiiRejectedError):
                _hide_rejected_identity(result)
        results.append(result)
    failed = sum(result["status"] != "ready" for result in results)
    return {
        "files_seen": len(results), "files_failed": failed,
        "chunk_count": sum(result.get("chunk_count", 0) for result in results),
        "total_tokens": sum(result.get("total_tokens", 0) for result in results),
        "status": "success" if not failed else "failed" if failed == len(results) else "partial_success",
        "results": results,
    }


def _embed_prepared_chunks(pool, chunks, settings, embedder, signature):
    contents = list(dict.fromkeys(chunk.content for chunk in chunks))
    reused = {}
    if getattr(settings, "embedding_reuse_enabled", False):
        hashes = [hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest() for content in contents]
        with stage("database_embedding_reuse"), pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            cur.execute(
                "SELECT c.content, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE md5(c.content) = ANY(%s) AND d.embedding_signature = %s "
                "AND c.embedding IS NOT NULL",
                (hashes, signature),
            )
            for row in cur.fetchall():
                if row["content"] not in contents:
                    continue
                try:
                    reused[row["content"]] = _validate_embeddings([row["embedding"]], 1, settings.embedding_dim)[0]
                except ValueError:
                    increment_metric("embedding_reuse_invalid")
    missing = [content for content in contents if content not in reused]
    increment_metric("embedding_reuse_hits", sum(chunk.content in reused for chunk in chunks))
    increment_metric("embedding_inputs", len(missing))
    if missing:
        vectors = _validate_embeddings(
            measured("embedding", embedder.embed_documents, missing), len(missing), settings.embedding_dim,
        )
        reused.update(zip(missing, vectors))
    return np.asarray([reused[chunk.content] for chunk in chunks], dtype=np.float32)


def _process_file(
    pool, file_path, settings, embedder, signature, full, *, source_path: str | None = None,
    result_info: dict | None = None,
) -> tuple[str, int]:
    """Prepare and atomically publish one document, skipping compatible duplicates."""
    doc = measured("extraction", load_file, Path(file_path))
    if doc is None:
        return ("unsupported", 0)
    if source_path is not None:
        doc.metadata["original_filename"] = Path(file_path).name
        doc.source_path = source_path
    doc = measured("pii_sanitization", sanitize_document, doc)
    if result_info is not None:
        result_info["source_path"] = doc.source_path
        filename = Path(file_path).name
        safe_filename = doc.metadata.get("original_filename", filename)
        result_info["file"] = "[sanitized source]" if doc.source_path.startswith("sanitized://") or safe_filename != filename else filename
        if "pii_sanitization" in doc.metadata:
            result_info["pii_sanitization"] = doc.metadata["pii_sanitization"]
    fingerprint = settings.processing_fingerprint
    doc.metadata.update(
        processing_fingerprint=fingerprint,
        ingestion_pipeline_version=settings.ingestion_pipeline_version,
    )
    digest = _document_content_hash(doc)
    with stage("database_preflight"), pool.connection() as conn:
        _require_compatible_embedding_space(conn, signature, settings.embedding_dim)
        cur = conn.cursor(row_factory=dict_row)
        existing = _read_document(cur, doc.source_path)
        if not full:
            reason = _skip_reason(cur, existing, digest, signature, fingerprint)
            if reason:
                return (reason, 0)
    chunks = measured("chunking", _prepare_chunks, doc, settings)
    if not chunks:
        return ("empty", 0)
    batch_size = getattr(settings, "chunk_insert_batch_size", 200)
    if batch_size <= 0:
        raise ValueError("CHUNK_INSERT_BATCH_SIZE must be positive.")
    vectors = _embed_prepared_chunks(pool, chunks, settings, embedder, signature)

    with stage("database_publication"), pool.connection() as conn:
        _require_compatible_embedding_space(conn, signature, settings.embedding_dim, lock=True)
        cur = conn.cursor(row_factory=dict_row)
        current = _read_document(cur, doc.source_path, lock=True)
        if not full:
            reason = _skip_reason(cur, current, digest, signature, fingerprint)
            if reason:
                return (reason, 0)
        initial_revision = (existing["id"], existing["version"]) if existing else None
        current_revision = (current["id"], current["version"]) if current else None
        conflict = (
            f"Document {doc.source_path!r} changed during ingestion. "
            "Retry ingestion to process the current version."
        )
        if initial_revision != current_revision:
            raise ValueError(conflict)

        if current:
            doc_id = current["id"]
            conn.execute(
                "UPDATE documents SET title = %s, doc_type = %s, content_hash = %s, "
                "embedding_signature = %s, url = %s, version = version + 1, "
                "updated_at = now(), metadata = %s WHERE id = %s",
                (
                    doc.title,
                    doc.doc_type,
                    digest,
                    signature,
                    doc.url,
                    json.dumps(doc.metadata),
                    doc_id,
                ),
            )
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        else:
            cur.execute(
                "INSERT INTO documents (source_path, url, title, doc_type, content_hash, "
                "embedding_signature, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (source_path) DO NOTHING RETURNING id",
                (
                    doc.source_path,
                    doc.url,
                    doc.title,
                    doc.doc_type,
                    digest,
                    signature,
                    json.dumps(doc.metadata),
                ),
            )
            inserted = cur.fetchone()
            if inserted is None:
                raise ValueError(conflict)
            doc_id = inserted["id"]

        query = (
            "INSERT INTO chunks (document_id, chunk_index, heading, content, "
            "token_count, url_anchor, embedding, embedding_model, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        rows = [
                (
                    doc_id,
                    chunk.index,
                    chunk.heading,
                    chunk.content,
                    chunk.token_count,
                    f"page={chunk.metadata['page_number']}" if "page_number" in chunk.metadata else None,
                    vector,
                    embedder.name,
                    json.dumps(chunk.metadata),
                )
            for chunk, vector in zip(chunks, vectors)
        ]
        if len(rows) == 1:
            conn.execute(query, rows[0])
        else:
            for start in range(0, len(rows), batch_size):
                cur.executemany(query, rows[start:start + batch_size])
                increment_metric("database_chunk_batches")

    return ("indexed", len(chunks))


def _require_matching_dim(settings, embedder) -> None:
    if embedder.dimension != settings.embedding_dim:
        raise ValueError(
            f"Provider dimension {embedder.dimension} != EMBEDDING_DIM "
            f"{settings.embedding_dim}. Fix config; rebuild in a separate database "
            "if changing the dimension."
        )


def _start_run(pool, trace_id: str | None = None) -> int:
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            "INSERT INTO ingestion_runs (files_failed, results, trace_id) "
            "VALUES (0, '[]'::jsonb, %s) RETURNING id",
            (trace_id,),
        )
        return cur.fetchone()["id"]


def _finish_run(pool, run_id, seen, changed, written, status, error=None, *, failed=0) -> None:
    with pool.connection() as conn:
        conn.execute(
            "UPDATE ingestion_runs SET finished_at = now(), files_seen = %s, "
            "files_changed = %s, chunks_written = %s, status = %s, error = %s, "
            "files_failed = %s WHERE id = %s",
            (seen, changed, written, status, error, failed, run_id),
        )


def _run_summary(results: list[dict]) -> dict:
    failed = sum(result["status"] not in {"indexed", "skipped", "duplicate"} for result in results)
    outcome = "success"
    if failed:
        outcome = "failed" if failed == len(results) else "partial_success"
    return {
        "files_seen": len(results),
        "files_changed": sum(result["status"] == "indexed" for result in results),
        "chunks_written": sum(result["chunks"] for result in results),
        "files_failed": failed,
        "status": outcome,
        "results": results,
    }


def _record_file_result(pool, run_id, result: dict) -> None:
    failed = result["status"] not in {"indexed", "skipped", "duplicate"}
    with pool.connection() as conn:
        conn.execute(
            "UPDATE ingestion_runs SET files_seen = files_seen + 1, "
            "files_changed = files_changed + %s, chunks_written = chunks_written + %s, "
            "files_failed = files_failed + %s, results = results || %s::jsonb WHERE id = %s",
            (int(result["status"] == "indexed"), result["chunks"], int(failed),
             json.dumps([result]), run_id),
        )


def status() -> dict:
    """Report indexed documents and the embedding signature(s) in use.

    Mixed, unknown, or mismatched signatures block ingestion and require an
    isolated rebuild or restoring the configuration of a healthy existing index.
    """
    settings = get_settings()
    current = settings.embedding_signature
    pool = get_pool()
    with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute("SELECT count(*) AS docs FROM documents")
        docs = cur.fetchone()["docs"]
        cur.execute("SELECT count(*) AS chunks FROM chunks")
        chunks = cur.fetchone()["chunks"]
        cur.execute(
            "SELECT COALESCE(embedding_signature, '(none)') AS signature, count(*) AS n "
            "FROM documents GROUP BY embedding_signature ORDER BY n DESC"
        )
        signatures = cur.fetchall()

    indexed = [s["signature"] for s in signatures if s["signature"] != "(none)"]
    unknown = any(s["signature"] == "(none)" for s in signatures)
    mixed = len(signatures) > 1
    stale = unknown or any(sig != current for sig in indexed)

    return {
        "documents": docs,
        "chunks": chunks,
        "current_signature": current,
        "indexed_signatures": [dict(s) for s in signatures],
        "mixed": mixed,
        "stale": stale,
        "unknown": unknown,
    }


def ingest(path: str | None = None, full: bool = False) -> dict:
    """Ingest every supported file under a directory (or a single file)."""
    settings = get_settings()
    root = path or settings.docs_source_dir
    return _collect_results(_ingest_events(iter_files(root), full, fail_fast=True))


def ingest_stream(
    paths: list[str], full: bool = False, *, source_paths: dict[str, str] | None = None
):
    """Generator that ingests files one at a time, yielding progress events.

    Yields dicts:
      {"phase": "start", "index": i, "total": n, "file": name}
      {"phase": "done",  "index": i, "total": n, "file": name,
       "status": indexed|skipped|duplicate|unsupported|empty|failed,
       "chunks": int, "error": optional str}

    File results are persisted before their done events are yielded.
    """
    yield from _ingest_events(paths, full, total=len(paths), source_paths=source_paths)


def _ingest_events(paths, full, *, total=None, source_paths=None, fail_fast=False):
    settings = get_settings()
    embedder = get_embedding_provider()
    _require_matching_dim(settings, embedder)
    signature = settings.embedding_signature

    pool = get_pool()
    trace_id = create_trace_id()
    run_id = _start_run(pool, trace_id)
    results: list[dict] = []
    current: dict | None = None
    terminal_status = None
    error = None
    try:
        for index, file_path in enumerate(paths, 1):
            name = Path(file_path).name
            current = {
                "file": name,
                "source_path": source_paths.get(file_path, str(file_path))
                if source_paths is not None else str(file_path),
                "status": "interrupted", "chunks": 0,
            }
            _safe_result_identity(current)
            yield {"phase": "start", "index": index, "total": total, "file": current["file"]}
            try:
                try:
                    file_size = Path(file_path).stat().st_size
                except OSError:
                    file_size = None
                with observation(
                    "ingestion.file",
                    trace_id=trace_id,
                    metadata={
                        "file_extension": Path(file_path).suffix.lower(),
                        "file_size_bytes": file_size,
                    },
                ) as file_span:
                    with capture_metrics() as metrics:
                        current["metrics"] = metrics
                        info = {}
                        try:
                            status_, written = _process_file(
                                pool, file_path, settings, embedder, signature, full,
                                source_path=source_paths[file_path] if source_paths is not None else None,
                                result_info=info,
                            )
                        finally:
                            current.update(info)
                    if file_span is not None:
                        file_span.update(output={"status": status_, "chunks_written": written})
                current.update(status=status_, chunks=written)
                if status_ in {"empty", "unsupported"}:
                    current["error"] = "No indexable content or unsupported file format."
            except Exception as exc:  # noqa: BLE001
                current.update(status="failed", error=str(exc) or type(exc).__name__)
                if isinstance(exc, PiiRejectedError):
                    _hide_rejected_identity(current)
                if fail_fast:
                    raise
            result = current
            results.append(result)
            current = None
            _record_file_result(pool, run_id, result)
            yield {"phase": "done", "index": index, "total": total, **result}
    except BaseException as exc:
        completed = current is None and total is not None and len(results) == total
        if not isinstance(exc, GeneratorExit) or not completed:
            terminal_status = "failed" if isinstance(exc, Exception) else "interrupted"
            error = str(exc) or type(exc).__name__
        if current is not None:
            current.update(status=terminal_status, error=error)
            results.append(current)
            _record_file_result(pool, run_id, current)
        raise
    finally:
        summary = _run_summary(results)
        if summary["files_failed"] and error is None:
            error = f"{summary['files_failed']} file(s) failed or were rejected."
        _finish_run(
            pool, run_id, summary["files_seen"], summary["files_changed"],
            summary["chunks_written"], terminal_status or summary["status"], error,
            failed=summary["files_failed"],
        )


def _collect_results(events) -> dict:
    results = []
    with closing(events):
        for event in events:
            if event["phase"] == "done":
                results.append({
                    key: value for key, value in event.items()
                    if key not in {"phase", "index", "total"}
                })
    return _run_summary(results)


def ingest_files(paths: list[str], full: bool = False) -> dict:
    """Ingest an explicit list of files (used by the upload UI).

    Returns a summary plus a per-file breakdown so the UI can show each result.
    """
    return _collect_results(ingest_stream(paths, full))
