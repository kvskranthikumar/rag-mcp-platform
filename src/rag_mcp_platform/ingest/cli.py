"""Command-line entry point: init-db, ingest, search."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-mcp-platform",
        description="Manage a document RAG index and MCP platform.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Create the database schema and indexes.")
    p_init.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables first (required when changing embedding dimension).",
    )

    p_ingest = sub.add_parser("ingest", help="Ingest/re-index documents.")
    p_ingest.add_argument("--path", default=None, help="File or directory (default: DOCS_SOURCE_DIR).")
    p_ingest.add_argument(
        "--full", action="store_true", help="Re-index all files, ignoring content hashes."
    )

    p_inspect = sub.add_parser("inspect", help="Preview extraction and chunk quality without indexing.")
    p_inspect.add_argument("--path", default=None, help="File or directory (default: DOCS_SOURCE_DIR).")
    p_inspect.add_argument("--preview", action="store_true", help="Include chunk text in the report.")
    p_evaluate = sub.add_parser("evaluate", help="Run an offline extraction-quality manifest.")
    p_evaluate.add_argument("manifest", help="Path to a version 1 JSON evaluation manifest.")

    p_audit = sub.add_parser("audit", help="Read-only index audit; no cleanup is performed.")
    p_audit.add_argument("--pii", action="store_true", help="Also scan stored content for supported PII.")
    p_reconcile = sub.add_parser("reconcile", help="Preview missing local sources without deleting anything.")
    p_reconcile.add_argument("--path", required=True, help="Existing directory to reconcile.")
    for action in ("rename", "delete"):
        maintenance = sub.add_parser(action, help=f"Preview document {action}; --apply enables writes.")
        maintenance.add_argument("document_id")
        maintenance.add_argument("--expected-version", type=int, required=True)
        maintenance.add_argument("--apply", action="store_true")
        if action == "rename":
            maintenance.add_argument("--to", required=True)

    p_search = sub.add_parser("search", help="Debug: run a hybrid search query.")
    p_search.add_argument("query", help="The search query.")
    p_search.add_argument("-k", type=int, default=None, help="Number of results.")

    p_ask = sub.add_parser("ask", help="Debug: retrieve + generate an answer via the LLM.")
    p_ask.add_argument("query", help="The question.")
    p_ask.add_argument("-k", type=int, default=None, help="Number of context chunks.")

    sub.add_parser("status", help="Show index stats and embedding-signature health.")

    args = parser.parse_args()

    if args.command == "init-db":
        from ..schema import init_db

        init_db(drop=args.drop)
        print("Schema initialized." + (" (dropped existing tables)" if args.drop else ""))

    elif args.command == "ingest":
        from .indexer import ingest

        result = ingest(path=args.path, full=args.full)
        print(json.dumps(result, indent=2))
        if result["status"] != "success":
            raise SystemExit(1)

    elif args.command == "inspect":
        from .indexer import inspect_sources

        result = inspect_sources(args.path, preview=args.preview)
        print(json.dumps(result, indent=2))
        if result["status"] != "success":
            raise SystemExit(1)

    elif args.command == "evaluate":
        from .evaluation import evaluate_manifest

        result = evaluate_manifest(args.manifest)
        print(json.dumps(result, indent=2))
        if result["status"] != "success":
            raise SystemExit(1)

    elif args.command in {"audit", "reconcile", "rename", "delete"}:
        from .admin import audit_index, maintain_document, reconcile_sources

        if args.command == "audit":
            result = audit_index(check_pii=args.pii)
        elif args.command == "reconcile":
            result = reconcile_sources(args.path)
        else:
            result = maintain_document(
                args.document_id, args.expected_version, rename_to=getattr(args, "to", None), apply=args.apply,
            )
        print(json.dumps(result, indent=2))

    elif args.command == "search":
        from ..core.search import search

        results = search(args.query, args.k)
        if not results:
            print("No results.")
            return
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r.score:.4f}  {r.citation()}")
            print(f"    {r.content[:200].strip()}...\n")

    elif args.command == "ask":
        from ..core.search import answer

        result = answer(args.query, args.k)
        print(result["answer"])
        print("\nSources:")
        for src in result["sources"]:
            print(f"  - {src['citation']}")

    elif args.command == "status":
        from .indexer import status

        s = status()
        print(f"Documents: {s['documents']}   Chunks: {s['chunks']}")
        print(f"Current embedding signature: {s['current_signature']}")
        print("Indexed signatures:")
        for row in s["indexed_signatures"]:
            print(f"  - {row['signature']}: {row['n']} document(s)")
        if s["mixed"]:
            print("\nWARNING: index contains MULTIPLE embedding signatures (mixed "
                "vector spaces). Ingestion is blocked. Rebuild in a separate database.")
        elif s["stale"]:
            print("\nWARNING: indexed signatures are unknown or differ from current config. "
                "Restore the matching configuration or rebuild in a separate database.")
        else:
            print("\nOK: single, current embedding space.")


if __name__ == "__main__":
    main()
