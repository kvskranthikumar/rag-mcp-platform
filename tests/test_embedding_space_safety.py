import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.chunker import Chunk
from rag_mcp_platform.ingest.indexer import _process_file, _require_compatible_embedding_space, status
from rag_mcp_platform.ingest.loaders import LoadedDoc


class EmbeddingSpaceGuardTests(unittest.TestCase):
    def connection(self, column=("vector(2)",), incompatible=None):
        conn = MagicMock()

        def execute(query, parameters=None):
            result = MagicMock()
            if "pg_attribute" in query:
                result.fetchone.return_value = column
            elif "IS DISTINCT FROM" in query:
                result.fetchone.return_value = incompatible
            return result

        conn.execute.side_effect = execute
        return conn

    def test_compatible_or_empty_index_is_accepted(self):
        conn = self.connection()
        _require_compatible_embedding_space(conn, "test:2", 2)
        query = conn.execute.call_args
        self.assertIn("IS DISTINCT FROM %s", query.args[0])
        self.assertEqual(query.args[1], ("test:2",))
        self.assertEqual(conn.execute.call_count, 2)

    def test_different_and_unknown_signatures_are_rejected(self):
        for signature in ("old-model:2", None, ""):
            with self.subTest(signature=signature):
                conn = self.connection(incompatible=(signature,))
                with self.assertRaisesRegex(ValueError, "In-place embedding-space changes are blocked"):
                    _require_compatible_embedding_space(conn, "test:2", 2)

    def test_schema_dimension_is_checked_before_signature(self):
        for column in (("vector(3)",), ("vector",), None):
            with self.subTest(column=column):
                conn = self.connection(column=column)
                with self.assertRaisesRegex(ValueError, "configuration requires vector"):
                    _require_compatible_embedding_space(conn, "test:2", 2)
                self.assertEqual(conn.execute.call_count, 1)

    def test_publication_lock_precedes_both_safety_checks(self):
        conn = self.connection()
        _require_compatible_embedding_space(conn, "test:2", 2, lock=True)
        calls = conn.execute.call_args_list
        self.assertIn("pg_advisory_xact_lock", calls[0].args[0])
        self.assertIn("pg_attribute", calls[1].args[0])
        self.assertIn("IS DISTINCT FROM", calls[2].args[0])

    def test_different_models_use_the_same_publication_lock(self):
        first = self.connection()
        second = self.connection()
        _require_compatible_embedding_space(first, "model-a:2", 2, lock=True)
        _require_compatible_embedding_space(second, "model-b:2", 2, lock=True)
        self.assertEqual(first.execute.call_args_list[0], second.execute.call_args_list[0])


class EmbeddingSpacePublicationTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            chunk_max_tokens=700, chunk_overlap_tokens=100, embedding_dim=2,
            processing_fingerprint="pipeline:1", ingestion_pipeline_version="1",
        )
        self.pool = MagicMock()
        self.conn = MagicMock()
        self.conn.cursor.return_value.fetchone.return_value = None
        self.signature = None
        self.column = "vector(2)"
        self.active_connections = 0
        self.events = []

        @contextmanager
        def connection():
            self.active_connections += 1
            try:
                yield self.conn
            finally:
                self.active_connections -= 1

        def execute(query, parameters=None):
            self.events.append(query)
            result = MagicMock()
            if "pg_attribute" in query:
                result.fetchone.return_value = (self.column,)
            elif "IS DISTINCT FROM" in query:
                result.fetchone.return_value = (self.signature,) if self.signature else None
            return result

        self.pool.connection.side_effect = connection
        self.conn.execute.side_effect = execute
        self.embedder = MagicMock()
        self.embedder.name = "test"
        self.embedder.embed_documents.side_effect = self.embed

    def embed(self, texts):
        self.assertEqual(self.active_connections, 0)
        return [[0.1, 0.2]]

    def process(self, *, full=False):
        document = LoadedDoc("guide.txt", "text", "Guide", "Updated text")
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer.chunk_text", return_value=[
                Chunk(index=0, content=document.text, heading=None, token_count=2)
            ]),
        ):
            return _process_file(
                self.pool, "guide.txt", self.settings, self.embedder, "test:2", full
            )

    def assert_no_document_writes(self):
        self.assertFalse(any(query.startswith(("INSERT", "UPDATE", "DELETE")) for query in self.events))
        queries = self.conn.cursor.return_value.execute.call_args_list
        self.assertFalse(any(call.args[0].startswith(("INSERT", "UPDATE", "DELETE")) for call in queries))

    def test_incompatible_index_fails_before_embedding(self):
        self.signature = "old-model:2"
        with self.assertRaisesRegex(ValueError, "different or unknown embedding signature"):
            self.process()
        self.embedder.embed_documents.assert_not_called()
        self.assert_no_document_writes()

    def test_force_reindex_cannot_bypass_space_guard(self):
        self.signature = "old-model:2"
        with self.assertRaisesRegex(ValueError, "including --full"):
            self.process(full=True)
        self.embedder.embed_documents.assert_not_called()
        self.assert_no_document_writes()

    def test_dimension_mismatch_fails_before_embedding(self):
        self.column = "vector(3)"
        with self.assertRaisesRegex(ValueError, "configuration requires vector"):
            self.process()
        self.embedder.embed_documents.assert_not_called()
        self.assert_no_document_writes()

    def test_competing_model_is_detected_after_embedding(self):
        def embed(texts):
            self.assertEqual(self.active_connections, 0)
            self.signature = "competing-model:2"
            return [[0.1, 0.2]]

        self.embedder.embed_documents.side_effect = embed
        with self.assertRaisesRegex(ValueError, "different or unknown embedding signature"):
            self.process()
        self.assertEqual(self.embedder.embed_documents.call_count, 1)
        self.assertTrue(any("pg_advisory_xact_lock" in query for query in self.events))
        self.assert_no_document_writes()
        self.assertEqual(self.active_connections, 0)

    def test_compatible_index_publishes_after_locked_recheck(self):
        self.conn.cursor.return_value.fetchone.side_effect = [
            None, None, None, None, {"id": "new-document"},
        ]
        self.assertEqual(self.process(), ("indexed", 1))
        lock_index = next(index for index, query in enumerate(self.events) if "pg_advisory" in query)
        chunk_index = next(index for index, query in enumerate(self.events) if "INSERT INTO chunks" in query)
        self.assertLess(lock_index, chunk_index)
        self.assertEqual(sum("IS DISTINCT FROM" in query for query in self.events), 2)


class EmbeddingSpaceStatusTests(unittest.TestCase):
    def report(self, signatures):
        pool = MagicMock()
        cursor = pool.connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.side_effect = [{"docs": 2}, {"chunks": 4}]
        cursor.fetchall.return_value = [
            {"signature": signature, "n": 1} for signature in signatures
        ]
        with (
            patch("rag_mcp_platform.ingest.indexer.get_settings", return_value=SimpleNamespace(
                embedding_signature="test:2"
            )),
            patch("rag_mcp_platform.ingest.indexer.get_pool", return_value=pool),
        ):
            return status()

    def test_unknown_signatures_are_not_reported_as_healthy(self):
        report = self.report(["(none)"])
        self.assertTrue(report["unknown"])
        self.assertTrue(report["stale"])

    def test_known_and_unknown_signatures_are_mixed(self):
        report = self.report(["test:2", "(none)"])
        self.assertTrue(report["mixed"])
        self.assertTrue(report["stale"])

    def test_matching_and_empty_indexes_are_healthy(self):
        for signatures in (["test:2"], []):
            with self.subTest(signatures=signatures):
                report = self.report(signatures)
                self.assertFalse(report["unknown"])
                self.assertFalse(report["mixed"])
                self.assertFalse(report["stale"])

    def test_single_different_signature_is_stale(self):
        report = self.report(["old-model:2"])
        self.assertTrue(report["stale"])
        self.assertFalse(report["mixed"])


if __name__ == "__main__":
    unittest.main()