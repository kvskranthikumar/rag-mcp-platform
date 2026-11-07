"""Read-only index audits and explicit, preview-first document maintenance."""

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from psycopg.rows import dict_row

from ..db import get_pool
from ..settings import get_settings
from .indexer import _EMBEDDING_SPACE_LOCK
from .loaders import LoadedDoc, iter_files
from .pii import PiiRejectedError, reject_pii


def _reference(source_path: str) -> str:
    return "source://" + hashlib.sha256(source_path.encode("utf-8")).hexdigest()


def audit_index(*, check_pii: bool = False) -> dict:
    settings = get_settings()
    findings = []
    signatures = set()
    with get_pool().connection() as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            "SELECT d.id, d.source_path, d.embedding_signature, d.metadata, "
            "count(c.id) AS chunk_count, "
            "count(c.id) FILTER (WHERE c.embedding IS NULL) AS missing_vectors, "
            "array_agg(DISTINCT vector_dims(c.embedding)) FILTER (WHERE c.embedding IS NOT NULL) AS dimensions "
            "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id"
        )
        for row in cur.fetchall():
            issues = []
            signature = row["embedding_signature"]
            signatures.add(signature)
            if not signature:
                issues.append("unknown_embedding_signature")
            elif signature != settings.embedding_signature:
                issues.append("stale_embedding_signature")
            if (row["metadata"] or {}).get("processing_fingerprint") != settings.processing_fingerprint:
                issues.append("stale_processing_fingerprint")
            if not row["chunk_count"]:
                issues.append("no_chunks")
            if row["missing_vectors"]:
                issues.append("missing_embeddings")
            if any(dimension != settings.embedding_dim for dimension in row["dimensions"] or []):
                issues.append("embedding_dimension_mismatch")
            if "-ingest-" in row["source_path"].replace("\\", "/"):
                issues.append("legacy_temporary_upload_identity")
            findings.append({
                "document_id": str(row["id"]), "source_ref": _reference(row["source_path"]),
                "chunk_count": row["chunk_count"], "issues": issues,
            })
        if check_pii:
            by_id = {finding["document_id"]: finding for finding in findings}
            cur.execute(
                "SELECT d.id, d.source_path, d.title, d.url, d.doc_type, d.metadata AS document_metadata, "
                "c.content, c.metadata AS chunk_metadata FROM documents d "
                "LEFT JOIN chunks c ON c.document_id = d.id"
            )
            while rows := cur.fetchmany(100):
                for row in rows:
                    document = LoadedDoc(
                        row["source_path"], row["doc_type"], row["title"] or "", row["content"] or "",
                        url=row["url"], metadata={
                            "document": row["document_metadata"] or {}, "chunk": row["chunk_metadata"] or {},
                        },
                    )
                    try:
                        reject_pii(document)
                    except PiiRejectedError as exc:
                        finding = by_id[str(row["id"])]
                        if "pii_policy_rejection" not in finding["issues"]:
                            finding["issues"].append("pii_policy_rejection")
                            finding["pii_error"] = str(exc)
    return {
        "documents": len(findings), "documents_with_issues": sum(bool(item["issues"]) for item in findings),
        "mixed_embedding_spaces": len(signatures) > 1, "pii_checked": check_pii,
        "findings": findings,
    }


def reconcile_sources(root: str) -> dict:
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise ValueError("Reconciliation requires an existing source directory.")
    present = {path.resolve() for path in iter_files(directory)}
    missing = []
    with get_pool().connection() as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        cur = conn.cursor(row_factory=dict_row)
        cur.execute("SELECT id, source_path, version FROM documents")
        for row in cur.fetchall():
            source = row["source_path"]
            if "://" in source:
                continue
            path = Path(source).resolve()
            if path.is_relative_to(directory) and path not in present:
                missing.append({"document_id": str(row["id"]), "version": row["version"], "source_ref": _reference(source)})
    return {"root": str(directory), "missing_sources": missing, "applied": False}


def maintain_document(document_id: str, expected_version: int, *, rename_to=None, apply=False) -> dict:
    if rename_to is not None:
        parsed = urlsplit(rename_to)
        if parsed.scheme and parsed.scheme != "upload" and not Path(rename_to).drive:
            raise ValueError("Rename targets must be local paths or upload:// identities.")
        if not rename_to.strip():
            raise ValueError("Rename target must not be blank.")
        reject_pii(LoadedDoc(rename_to, "text", rename_to, ""))
    with get_pool().connection() as conn:
        if apply:
            conn.execute("SELECT pg_advisory_xact_lock(%s, %s)", _EMBEDDING_SPACE_LOCK)
        else:
            conn.execute("SET TRANSACTION READ ONLY")
        cur = conn.cursor(row_factory=dict_row)
        query = "SELECT id, source_path, version FROM documents WHERE id = %s"
        cur.execute(query + (" FOR UPDATE" if apply else ""), (document_id,))
        document = cur.fetchone()
        if document is None:
            raise ValueError("Document does not exist.")
        if document["version"] != expected_version:
            raise ValueError("Document revision changed; review a fresh preview before applying.")
        if rename_to is not None and rename_to != document["source_path"]:
            cur.execute("SELECT id FROM documents WHERE source_path = %s", (rename_to,))
            if cur.fetchone():
                raise ValueError("Rename target already exists.")
        if apply:
            if rename_to is None:
                conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            else:
                conn.execute(
                    "UPDATE documents SET source_path = %s, version = version + 1, updated_at = now() WHERE id = %s",
                    (rename_to, document_id),
                )
                conn.execute(
                    "UPDATE chunks SET metadata = jsonb_set(metadata, '{source_path}', to_jsonb(%s::text)) "
                    "WHERE document_id = %s", (rename_to, document_id),
                )
    return {
        "document_id": document_id, "version": expected_version,
        "action": "delete" if rename_to is None else "rename", "applied": apply,
        "source_ref": _reference(document["source_path"]),
        "target_ref": _reference(rename_to) if rename_to is not None else None,
    }