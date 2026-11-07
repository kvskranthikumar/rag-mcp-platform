import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rag_mcp_platform.ingest.cli import main
from rag_mcp_platform.ingest.indexer import inspect_sources
from rag_mcp_platform.ingest.loaders import iter_files
from rag_mcp_platform.settings import Settings


class IngestionInspectionTests(unittest.TestCase):
    def test_preview_uses_real_preparation_without_database_or_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text("# Guide\n\n## Setup\n\nText for setup.", encoding="utf-8")
            with (
                patch("rag_mcp_platform.ingest.indexer.get_pool", side_effect=AssertionError("database accessed")),
                patch("rag_mcp_platform.ingest.indexer.get_embedding_provider", side_effect=AssertionError("provider accessed")),
            ):
                report = inspect_sources(str(path), preview=True)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["chunk_count"], 1)
        self.assertEqual(report["results"][0]["chunks"][0]["metadata"]["section_path"], ["Guide", "Setup"])

    def test_warnings_do_not_mark_valid_content_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.htm"
            path.write_text("<h1>Guide</h1><p>Content.</p>", encoding="utf-8")
            report = inspect_sources(str(path))
        self.assertEqual(report["status"], "success")
        self.assertIn("Missing source metadata", report["results"][0]["warnings"][0])
        self.assertNotIn("chunks", report["results"][0])

    def test_empty_files_are_reported_as_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text("", encoding="utf-8")
            report = inspect_sources(str(path))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["files_failed"], 1)

    def test_unsplittable_blocks_are_reported_per_file(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bad.md").write_text("```\n" + "x" * 100 + "\n```", encoding="utf-8")
            (Path(directory) / "good.md").write_text("Useful content.", encoding="utf-8")
            with patch("rag_mcp_platform.ingest.indexer.get_settings", return_value=Settings(
                _env_file=None, chunk_max_tokens=20, chunk_overlap_tokens=0,
            )), patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
                report = inspect_sources(directory)
        self.assertEqual(report["status"], "partial_success")
        self.assertEqual(report["files_failed"], 1)
        self.assertIn("Structured block exceeds", report["results"][0]["error"])

    def test_duplicate_content_is_flagged_within_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("one.md", "two.md"):
                (Path(directory) / name).write_text("Duplicate text.", encoding="utf-8")
            report = inspect_sources(directory)
        self.assertIn("Duplicate content", report["results"][1]["warnings"][0])

    def test_missing_and_unsupported_paths_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                list(iter_files(Path(directory) / "missing"))
            path = Path(directory) / "unsupported.bin"
            path.write_bytes(b"not a document")
            with self.assertRaisesRegex(ValueError, "Unsupported documentation"):
                list(iter_files(path))

    def test_cli_inspection_prints_report(self):
        output = StringIO()
        with (
            patch.object(sys, "argv", ["rag-mcp-platform", "inspect", "--path", "guide.md", "--preview"]),
            patch("rag_mcp_platform.ingest.indexer.inspect_sources", return_value={"status": "success"}) as inspect,
            redirect_stdout(output),
        ):
            main()
        inspect.assert_called_once_with("guide.md", preview=True)
        self.assertEqual(json.loads(output.getvalue())["status"], "success")


if __name__ == "__main__":
    unittest.main()