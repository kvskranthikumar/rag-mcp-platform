"""Streamlit ingestion UI.

A separate, hostable app for loading documents
into the vector store by drag-and-drop. It reuses the same ingestion core as the
CLI (``rag_mcp_platform.ingest.indexer``), so uploaded files are chunked, embedded via
the configured provider, and upserted into PostgreSQL/pgvector.

Run with:  ``rag-mcp-platform-ingest-ui``  (or ``streamlit run <this file>``)
"""

from __future__ import annotations

import tempfile
from contextlib import closing
from pathlib import Path

import streamlit as st

# Absolute imports so this works when Streamlit runs the file as a script
# (relative imports have no package context under `streamlit run`).
from rag_mcp_platform.db import check_connection
from rag_mcp_platform.ingest.indexer import ingest_stream, status
from rag_mcp_platform.ingest.loaders import SUPPORTED_EXTENSIONS
from rag_mcp_platform.ingest_ui.uploads import stage_uploads

st.set_page_config(page_title="Docs Ingestion", page_icon="📄", layout="wide")

UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)

# Streamlit's file_uploader already accepts drag-and-drop; this CSS turns the
# small default control into a large, obvious drop zone. The block-container
# rule removes the width cap that "wide" layout still applies.
_APP_CSS = """
<style>
[data-testid="stAppViewContainer"] .block-container {
    max-width: 100%;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1.25rem;
    /* Streamlit reserves ~10rem below the last element, which shows up as dead
       scrollable space at the bottom of a short page. */
    padding-bottom: 1.5rem;
}
/* Streamlit's fixed header bar reserves ~3.75rem of empty space above the
   content; shrink it so only the (transparent) toolbar strip remains. */
[data-testid="stHeader"] {
    height: 1.75rem;
    background: transparent;
}
.app-header {
    margin: 0 0 0.5rem 0;
}
.app-header .app-summary {
    margin: 0.15rem 0 0 0;
    font-size: 0.92rem;
    line-height: 1.35;
    opacity: 0.75;
    text-align: left;
}
[data-testid="stFileUploaderDropzone"] {
    min-height: 200px;
    border: 2px dashed #6c8cff;
    border-radius: 16px;
    background: rgba(108, 140, 255, 0.06);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: background 0.15s ease, border-color 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(108, 140, 255, 0.14);
    border-color: #4f6ef2;
}
/* Disabled uploader: no drop / browse / file-remove while ingesting. */
div[data-testid="stFileUploader"]:has(input:disabled) [data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploader"]:has([aria-disabled="true"]) [data-testid="stFileUploaderDropzone"] {
    pointer-events: none !important;
    opacity: 0.55;
    cursor: not-allowed !important;
    background: rgba(128, 128, 128, 0.08);
    border-color: #b0b8c8;
}
div[data-testid="stFileUploader"]:has(input:disabled) [data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(128, 128, 128, 0.08);
    border-color: #b0b8c8;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    align-items: center;
    text-align: center;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 1.15rem;
    font-weight: 600;
}
.stAppDeployButton,
[data-testid="stAppDeployButton"],
div[data-testid="stToolbar"] .stDeployButton {
    display: none !important;
}
[data-testid="stHeaderActionElements"],
[data-testid="StyledLinkIconContainer"],
div[data-testid="stHeadingWithActionElements"] a,
div[data-testid="stHeading"] a {
    display: none !important;
}
</style>
"""


_STATUS_BADGE = {
    "pending": "⏳ Pending",
    "processing": "🔄 Processing…",
    "indexed": "✅ Indexed",
    "duplicate": "♻️ Duplicate",
    "skipped": "⏭️ Unchanged",
    "unsupported": "⚠️ Unsupported",
    "empty": "⚠️ Empty",
    "failed": "❌ Failed",
    "interrupted": "Interrupted",
}


def _grid_rows(states: list[dict]) -> list[dict]:
    """One line-item per file, with status as its own column."""
    rows: list[dict] = []
    for i, state in enumerate(states, start=1):
        status_key = state["status"]
        detail = ""
        if state.get("error"):
            detail = state["error"]
        elif status_key == "duplicate":
            detail = "Identical content already in the index"
        elif status_key == "skipped":
            detail = "Unchanged since last ingest"
        rows.append(
            {
                "#": i,
                "File": state["file"],
                "Status": _STATUS_BADGE.get(status_key, status_key),
                "Chunks": int(state.get("chunks") or 0),
                "Details": detail,
            }
        )
    return rows


def _render_grid(container, states: list[dict]) -> None:
    """Render the file line-item grid into a placeholder container."""
    with container.container():
        st.dataframe(
            _grid_rows(states),
            hide_index=True,
            use_container_width=True,
            height=min(480, 37 * (len(states) + 1) + 16),
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "File": st.column_config.TextColumn("File", width="large"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Chunks": st.column_config.NumberColumn("Chunks", format="%d", width="small"),
                "Details": st.column_config.TextColumn("Details", width="large"),
            },
        )


def _render_results(states: list[dict], error: str | None = None) -> None:
    """Render the final grid + summary for a completed ingestion run."""
    total = len(states)
    n_indexed = sum(1 for s in states if s["status"] == "indexed")
    n_dup = sum(1 for s in states if s["status"] == "duplicate")
    n_skip = sum(1 for s in states if s["status"] == "skipped")
    n_chunks = sum(s["chunks"] for s in states)
    n_failed = sum(s["status"] not in {"indexed", "skipped", "duplicate"} for s in states)

    st.subheader("Last ingestion result", anchor=False)
    _render_grid(st.empty(), states)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", total)
    c2.metric("Indexed", n_indexed)
    c3.metric("Chunks", n_chunks)
    c4.metric("Failed", n_failed)

    if n_dup or n_skip:
        st.info(
            f"Ignored {n_dup + n_skip} file(s) already in the index "
            f"({n_dup} duplicate content, {n_skip} unchanged)."
        )
    if error:
        st.error(f"Ingestion stopped: {error}")
    elif n_failed == total and total:
        st.error(f"All {total} file(s) failed or were rejected. See the grid above for details.")
    elif n_failed:
        st.warning(f"{n_failed} file(s) failed or were rejected. See the grid above for details.")
    else:
        st.success("Ingestion complete. Drop new files above to ingest more.")


def _render_status(*, disabled: bool = False) -> None:
    st.subheader("Index status", anchor=False)
    if st.button("Refresh status", disabled=disabled):
        st.session_state.pop("_status", None)

    if "_status" not in st.session_state:
        try:
            st.session_state["_status"] = status()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read index status: {exc}")
            return

    info = st.session_state["_status"]
    col1, col2 = st.columns(2)
    col1.metric("Documents", info["documents"])
    col2.metric("Chunks", info["chunks"])

    if info["mixed"]:
        st.warning(
            "Index contains multiple embedding signatures (mixed vector spaces). "
            "Ingestion is blocked. Rebuild in a separate database."
        )
    elif info["stale"]:
        st.warning(
            "Indexed embedding signatures are unknown or differ from current config. "
            "Restore the matching configuration or rebuild in a separate database."
        )


def _run_ingest(files, full: bool) -> None:
    """Save uploads to a temp dir, stream ingest, then reset the UI."""
    with st.spinner("Checking database connection…"):
        ok, message = check_connection()
    if not ok:
        st.session_state["ingesting"] = False
        st.error(f"Cannot start ingestion — database connection failed.\n\n{message}")
        return

    st.success(f"Database connection OK. Ingesting {len(files)} file(s)…")

    with tempfile.TemporaryDirectory(prefix="rag-mcp-platform-ingest-") as tmp:
        try:
            source_paths = stage_uploads(files, Path(tmp))
            saved = list(source_paths)
        except Exception as exc:  # noqa: BLE001
            st.session_state["ingesting"] = False
            st.error(f"Failed to save uploads: {exc}")
            return

        states = [{"file": Path(p).name, "status": "pending", "chunks": 0} for p in saved]
        total = len(states)
        index_by_name: dict[str, int] = {s["file"]: i for i, s in enumerate(states)}

        st.subheader("Ingestion progress", anchor=False)
        progress = st.progress(0.0, text=f"0 / {total} files")
        grid = st.empty()
        _render_grid(grid, states)

        done = 0
        st.session_state.pop("last_run_error", None)
        try:
            with closing(ingest_stream(saved, full=full, source_paths=source_paths)) as events:
                for event in events:
                    idx = index_by_name.get(event["file"], event["index"] - 1)
                    states[idx]["file"] = event["file"]
                    if event["phase"] == "start":
                        states[idx]["status"] = "processing"
                    else:
                        states[idx]["file"] = event["file"]
                        states[idx]["status"] = event["status"]
                        states[idx]["chunks"] = event.get("chunks", 0)
                        if event.get("error"):
                            states[idx]["error"] = event["error"]
                        done += 1
                        progress.progress(done / total, text=f"{done} / {total} files")
                    _render_grid(grid, states)
        except Exception as exc:  # noqa: BLE001
            message = str(exc) or type(exc).__name__
            st.session_state["last_run_error"] = message
            for state in states:
                if state["status"] in {"pending", "processing"}:
                    state.update(status="interrupted", error=message)
            st.error(f"Ingestion stopped due to an error: {message}")

        st.session_state["last_results"] = states
        st.session_state.pop("_status", None)
        st.session_state["ingesting"] = False
        st.session_state["uploader_key"] += 1

    st.rerun()


def main() -> None:
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    st.subheader("Document Ingestion", anchor=False)
    st.markdown(
        (
            '<div class="app-header">'
            '<p class="app-summary">'
            "Drag and drop documentation files (or click to browse). "
            f"Supported types: {', '.join(UPLOAD_TYPES)}"
            "</p></div>"
        ),
        unsafe_allow_html=True,
    )

    # A changing key lets us fully reset the uploader (clear dropped files) after
    # an ingestion completes, so the drop area is empty for the next batch.
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("ingesting", False)
    ingesting = bool(st.session_state["ingesting"])

    files = st.file_uploader(
        "Drag & drop files here",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Drop one or more documents, or click Browse files.",
        key=f"uploader_{st.session_state['uploader_key']}",
        disabled=ingesting,
    )
    if files and not ingesting:
        # Fresh drop -> reset the UI (discard the previous run's results/grid).
        st.session_state.pop("last_results", None)
        st.session_state.pop("last_run_error", None)
        st.success(f"{len(files)} file(s) ready to ingest.")
    elif ingesting:
        st.info("Ingestion in progress. Dropping more files and other actions are paused.")

    full = st.checkbox(
        "Force re-index (ignore unchanged content / re-embed)",
        help="Reprocess using the current embedding space. Model changes require a separate database rebuild.",
        key="force_reindex",
        disabled=ingesting,
    )

    n_files = len(files) if files else 0
    if ingesting:
        ingest_label = f"Ingesting {n_files} file(s)…" if n_files else "Ingesting…"
    elif files:
        ingest_label = f"Ingest {n_files} file(s)"
    else:
        ingest_label = "Ingest"

    if st.button(
        ingest_label,
        type="primary",
        disabled=ingesting or not files,
        use_container_width=True,
    ):
        # Rerun once with widgets disabled, then start ingest on that next pass.
        st.session_state["ingesting"] = True
        st.rerun()

    if ingesting:
        if not files:
            st.session_state["ingesting"] = False
            st.error("Ingestion could not start — the file list is empty.")
        else:
            _run_ingest(files, full)

    # Show the previous run's grid + summary (persists across the uploader reset,
    # cleared automatically as soon as new files are dropped above).
    if not ingesting and not files and st.session_state.get("last_results"):
        _render_results(st.session_state["last_results"], st.session_state.get("last_run_error"))

    st.divider()
    _render_status(disabled=ingesting)


# Streamlit executes this module top-to-bottom on every run.
main()
