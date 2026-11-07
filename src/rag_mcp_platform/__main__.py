"""Allow ``python -m rag_mcp_platform <command>`` when the console script is not on PATH."""

from rag_mcp_platform.ingest.cli import main

if __name__ == "__main__":
    main()
