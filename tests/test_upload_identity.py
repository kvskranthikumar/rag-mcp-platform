import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.indexer import _process_file, ingest_stream
from rag_mcp_platform.ingest_ui.uploads import stage_uploads


def upload(name, content):
    uploaded = BytesIO(content)
    uploaded.name = name
    return uploaded


class UploadIdentityTests(unittest.TestCase):
    def setUp(self):
        guard = patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space")
        guard.start()
        self.addCleanup(guard.stop)

    def test_identity_survives_temporary_directory_changes(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            original = stage_uploads([upload("Guide #1.md", b"original")], Path(first))
            updated = stage_uploads([upload("Guide #1.md", b"changed")], Path(second))
            self.assertNotEqual(list(original), list(updated))
            self.assertEqual(list(original.values()), ["upload://Guide%20%231.md"])
            self.assertEqual(list(original.values()), list(updated.values()))

    def test_duplicate_names_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Duplicate upload filename"):
                stage_uploads(
                    [upload("Guide.md", b"one"), upload("Guide.md", b"two")], Path(directory)
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_paths_are_reduced_to_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = stage_uploads([upload(r"..\guide.md", b"text")], Path(directory))
            self.assertEqual(list(staged), [str(Path(directory) / "guide.md")])
            self.assertEqual(list(staged.values()), ["upload://guide.md"])

    def test_changed_upload_updates_existing_document(self):
        self.check_existing_upload(unchanged=False)

    def test_unchanged_upload_is_skipped(self):
        self.check_existing_upload(unchanged=True)

    def check_existing_upload(self, *, unchanged):
        with tempfile.TemporaryDirectory() as directory:
            staged = stage_uploads([upload("guide.txt", b"updated text")], Path(directory))
            file_path, source_path = next(iter(staged.items()))
            pool = MagicMock()
            conn = pool.connection.return_value.__enter__.return_value
            cursor = conn.cursor.return_value
            existing = {
                "id": "existing-document",
                "version": 1,
                "content_hash": hashlib.sha256(b"updated text").hexdigest()
                if unchanged else "old-hash",
                "embedding_signature": "test:2",
                "metadata": {"processing_fingerprint": "pipeline:1"},
            }
            cursor.fetchone.side_effect = [existing, existing]
            embedder = MagicMock(name="embedder")
            embedder.name = "test"
            embedder.embed_documents.return_value = [[0.1, 0.2]]
            settings = SimpleNamespace(
                chunk_max_tokens=700, chunk_overlap_tokens=100,
                processing_fingerprint="pipeline:1", ingestion_pipeline_version="1",
                embedding_dim=2,
            )
            chunk = SimpleNamespace(index=0, heading=None, content="updated text", token_count=2, metadata={})
            with patch("rag_mcp_platform.ingest.indexer.chunk_text", return_value=[chunk]):
                result = _process_file(
                    pool, file_path, settings, embedder, "test:2", False,
                    source_path=source_path,
                )
            self.assertEqual(cursor.execute.call_args_list[0].args[1], ("upload://guide.txt",))
            if unchanged:
                self.assertEqual(result, ("skipped", 0))
                embedder.embed_documents.assert_not_called()
                conn.execute.assert_not_called()
            else:
                self.assertEqual(result, ("indexed", 1))
                update = conn.execute.call_args_list[0]
                self.assertTrue(update.args[0].startswith("UPDATE documents"))
                self.assertEqual(update.args[1][-1], "existing-document")
                self.assertEqual(json.loads(update.args[1][-2]), {
                    "original_filename": "guide.txt",
                    "processing_fingerprint": "pipeline:1",
                    "ingestion_pipeline_version": "1",
                })
                self.assertFalse(any(
                    call.args[0].startswith("INSERT INTO documents")
                    for call in cursor.execute.call_args_list
                ))

    def test_stream_passes_logical_identity_to_indexer(self):
        settings = SimpleNamespace(embedding_dim=2, embedding_signature="test:2")
        embedder = SimpleNamespace(dimension=2)
        with (
            patch("rag_mcp_platform.ingest.indexer.get_settings", return_value=settings),
            patch("rag_mcp_platform.ingest.indexer.get_embedding_provider", return_value=embedder),
            patch("rag_mcp_platform.ingest.indexer.get_pool"),
            patch("rag_mcp_platform.ingest.indexer._start_run"),
            patch("rag_mcp_platform.ingest.indexer._finish_run"),
            patch("rag_mcp_platform.ingest.indexer._process_file", return_value=("indexed", 1)) as process,
        ):
            events = list(ingest_stream(
                ["temp/guide.txt"], source_paths={"temp/guide.txt": "upload://guide.txt"}
            ))
        self.assertEqual(process.call_args.kwargs, {"source_path": "upload://guide.txt", "result_info": {}})
        self.assertEqual(events[-1]["status"], "indexed")


if __name__ == "__main__":
    unittest.main()