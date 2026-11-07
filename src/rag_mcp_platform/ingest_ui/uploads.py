"""Stage uploads separately from their persistent document identities."""

from pathlib import Path, PureWindowsPath
from urllib.parse import quote


def stage_uploads(files, dest_dir: Path) -> dict[str, str]:
    """Return staged paths mapped to stable, filename-based upload identities."""
    uploads = []
    seen: set[str] = set()
    for uploaded in files:
        filename = PureWindowsPath(uploaded.name).name
        if not filename or filename in {".", ".."} or ":" in filename:
            raise ValueError(f"Invalid upload filename: {uploaded.name!r}")
        if filename.casefold() in seen:
            raise ValueError(f"Duplicate upload filename: {filename}. Rename one file and retry.")
        seen.add(filename.casefold())
        uploads.append((uploaded, filename))

    dest_dir.mkdir(parents=True, exist_ok=True)
    source_paths: dict[str, str] = {}
    for uploaded, filename in uploads:
        target = dest_dir / filename
        target.write_bytes(uploaded.getbuffer())
        source_paths[str(target)] = f"upload://{quote(filename, safe='')}"
    return source_paths