import hashlib
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.chunker import Chunk
from rag_mcp_platform.ingest.indexer import _process_file
from rag_mcp_platform.ingest.loaders import LoadedDoc


class IngestionTransactionTests(unittest.TestCase):
    def setUp(self):
        guard = patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space")
        guard.start()
        self.addCleanup(guard.stop)
        self.settings = SimpleNamespace(
            chunk_max_tokens=700, chunk_overlap_tokens=100, embedding_dim=2,
            processing_fingerprint="pipeline:1", ingestion_pipeline_version="1",
        )
        self.original = {
            "id": "document", "version": 1, "content_hash": "old-content",
            "embedding_signature": "test:2",
            "metadata": {"processing_fingerprint": "pipeline:1"},
        }
        self.preflight = MagicMock()
        self.publication = MagicMock()
        self.preflight.cursor.return_value.fetchone.return_value = dict(self.original)
        self.publication.cursor.return_value.fetchone.return_value = dict(self.original)
        self.pool = MagicMock()
        self.active_connections = 0
        self.connections_used = 0
        self.transaction_errors = []

        @contextmanager
        def connection():
            conn = [self.preflight, self.publication][self.connections_used]
            self.connections_used += 1
            self.active_connections += 1
            try:
                yield conn
            except BaseException as exc:
                self.transaction_errors.append(exc)
                raise
            finally:
                self.active_connections -= 1

        self.pool.connection.side_effect = connection
        self.embedder = MagicMock()
        self.embedder.name = "test"
        self.embedder.embed_documents.side_effect = self.embed

    def embed(self, texts):
        self.assertEqual(self.active_connections, 0)
        self.assertEqual(self.connections_used, 1)
        return [[0.1, 0.2]]

    def chunk(self, *args):
        self.assertEqual(self.active_connections, 0)
        return [Chunk(index=0, content="Updated text", heading=None, token_count=2)]

    def process(self, *, full=False):
        document = LoadedDoc("guide.txt", "text", "Guide", "Updated text")
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer.chunk_text", side_effect=self.chunk),
        ):
            return _process_file(
                self.pool, "guide.txt", self.settings, self.embedder, "test:2", full
            )

    def test_preparation_does_not_hold_a_pooled_connection(self):
        self.assertEqual(self.process(), ("indexed", 1))
        self.assertEqual(self.active_connections, 0)
        self.assertEqual(self.connections_used, 2)
        self.preflight.execute.assert_not_called()
        query = self.publication.cursor.return_value.execute.call_args_list[0]
        self.assertTrue(query.args[0].endswith("FOR UPDATE"))
        self.assertEqual(len(self.publication.execute.call_args_list), 3)

    def test_provider_failure_never_opens_publication_transaction(self):
        self.embedder.embed_documents.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.process()
        self.assertEqual(self.connections_used, 1)
        self.assertEqual(self.active_connections, 0)
        self.publication.execute.assert_not_called()

    def test_invalid_embedding_never_opens_publication_transaction(self):
        self.embedder.embed_documents.side_effect = None
        self.embedder.embed_documents.return_value = []
        with self.assertRaisesRegex(ValueError, "Expected 1 embedding"):
            self.process()
        self.assertEqual(self.connections_used, 1)
        self.publication.execute.assert_not_called()

    def test_competing_update_is_not_overwritten(self):
        self.publication.cursor.return_value.fetchone.return_value = dict(self.original, version=2)
        with self.assertRaisesRegex(ValueError, "changed during ingestion"):
            self.process()
        self.publication.execute.assert_not_called()
        self.assertEqual(self.active_connections, 0)

    def test_deleted_document_is_not_recreated(self):
        self.publication.cursor.return_value.fetchone.return_value = None
        with self.assertRaisesRegex(ValueError, "changed during ingestion"):
            self.process()
        self.publication.execute.assert_not_called()

    def test_recreated_document_is_not_overwritten(self):
        self.publication.cursor.return_value.fetchone.return_value = dict(self.original, id="replacement")
        with self.assertRaisesRegex(ValueError, "changed during ingestion"):
            self.process()
        self.publication.execute.assert_not_called()

    def test_matching_competing_update_is_skipped(self):
        self.publication.cursor.return_value.fetchone.return_value = dict(
            self.original, version=2,
            content_hash=hashlib.sha256(b"Updated text").hexdigest(),
        )
        self.assertEqual(self.process(), ("skipped", 0))
        self.publication.execute.assert_not_called()

    def test_force_reindex_does_not_overwrite_competing_update(self):
        self.publication.cursor.return_value.fetchone.return_value = dict(self.original, version=2)
        with self.assertRaisesRegex(ValueError, "changed during ingestion"):
            self.process(full=True)
        self.publication.execute.assert_not_called()

    def test_new_document_appearing_during_embedding_is_not_overwritten(self):
        self.preflight.cursor.return_value.fetchone.return_value = None
        with self.assertRaisesRegex(ValueError, "changed during ingestion"):
            self.process()
        self.publication.execute.assert_not_called()

    def test_duplicate_appearing_during_embedding_is_skipped(self):
        self.preflight.cursor.return_value.fetchone.return_value = None
        self.publication.cursor.return_value.fetchone.side_effect = [
            None, {"source_path": "another.txt"},
        ]
        self.assertEqual(self.process(), ("duplicate", 0))
        self.publication.execute.assert_not_called()

    def test_concurrent_insert_conflict_is_reported_without_replacement(self):
        self.preflight.cursor.return_value.fetchone.return_value = None
        self.publication.cursor.return_value.fetchone.return_value = None
        with self.assertRaisesRegex(ValueError, "changed during ingestion"):
            self.process()
        query = self.publication.cursor.return_value.execute.call_args_list[-1]
        self.assertIn("ON CONFLICT (source_path) DO NOTHING", query.args[0])
        self.publication.execute.assert_not_called()

    def test_publication_failure_reaches_transaction_context(self):
        self.publication.execute.side_effect = [None, None, RuntimeError("chunk insert failed")]
        with self.assertRaisesRegex(RuntimeError, "chunk insert failed"):
            self.process()
        self.assertEqual(self.active_connections, 0)
        self.assertEqual(len(self.transaction_errors), 1)
        self.assertEqual(str(self.transaction_errors[0]), "chunk insert failed")

    def test_bulk_insert_failure_reaches_transaction_context(self):
        self.chunk = lambda *args: [
            Chunk(0, None, "First text", 2), Chunk(1, None, "Second text", 2),
        ]
        self.embedder.embed_documents.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
        self.publication.cursor.return_value.executemany.side_effect = RuntimeError("bulk insert failed")
        with self.assertRaisesRegex(RuntimeError, "bulk insert failed"):
            self.process()
        self.assertEqual(self.active_connections, 0)
        self.assertEqual(str(self.transaction_errors[0]), "bulk insert failed")


if __name__ == "__main__":
    unittest.main()