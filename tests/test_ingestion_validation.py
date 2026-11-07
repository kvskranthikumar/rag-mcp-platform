import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from rag_mcp_platform.ingest.chunker import Chunk
from rag_mcp_platform.ingest.indexer import _process_file
from rag_mcp_platform.ingest.loaders import LoadedDoc


class IngestionValidationTests(unittest.TestCase):
    def setUp(self):
        guard = patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space")
        guard.start()
        self.addCleanup(guard.stop)
        self.settings = SimpleNamespace(
            chunk_max_tokens=700, chunk_overlap_tokens=100, embedding_dim=2,
            processing_fingerprint="pipeline:1", ingestion_pipeline_version="1",
        )
        self.pool = MagicMock()
        self.conn = self.pool.connection.return_value.__enter__.return_value
        self.cursor = self.conn.cursor.return_value
        self.existing = {"id": "existing-document", "version": 1}
        self.cursor.fetchone.return_value = self.existing
        self.embedder = MagicMock()
        self.embedder.name = "test"
        self.embedder.embed_documents.return_value = [[0.1, 0.2]]

    def process(self, chunks):
        document = LoadedDoc("guide.txt", "text", "Guide", "changed content")
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer.chunk_text", return_value=chunks),
        ):
            return _process_file(
                self.pool, "guide.txt", self.settings, self.embedder, "test:2", True
            )

    def chunk(self, *, index=0, content="Valid text", token_count=2):
        return Chunk(index=index, content=content, token_count=token_count, heading=None)

    def assert_no_publication(self):
        self.conn.execute.assert_not_called()
        self.cursor.execute.assert_called_once()
        self.assertTrue(self.cursor.execute.call_args.args[0].startswith("SELECT"))

    def test_empty_chunk_list_preserves_existing_document(self):
        self.assertEqual(self.process([]), ("empty", 0))
        self.assert_no_publication()
        self.embedder.embed_documents.assert_not_called()

    def test_empty_chunk_list_does_not_create_new_document(self):
        self.cursor.fetchone.return_value = None
        self.assertEqual(self.process([]), ("empty", 0))
        self.assert_no_publication()

    def test_invalid_chunks_are_rejected_before_embedding(self):
        cases = (
            (self.chunk(content=" \n\t"), "content"),
            (self.chunk(content=None), "content"),
            (self.chunk(index=1), "index"),
            (self.chunk(index=True), "index"),
            (self.chunk(token_count=0), "token count"),
            (self.chunk(token_count=-1), "token count"),
            (self.chunk(token_count=701), "token count"),
            (self.chunk(token_count=True), "token count"),
        )
        for chunk, message in cases:
            with self.subTest(chunk=chunk):
                self.cursor.reset_mock()
                with self.assertRaisesRegex(ValueError, message):
                    self.process([chunk])
                self.assert_no_publication()
                self.embedder.embed_documents.assert_not_called()

    def test_duplicate_chunk_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sequential index 1"):
            self.process([self.chunk(), self.chunk()])
        self.assert_no_publication()
        self.embedder.embed_documents.assert_not_called()

    def test_wrong_embedding_counts_are_rejected(self):
        for vectors in ([], [[0.1, 0.2], [0.3, 0.4]], None):
            with self.subTest(vectors=vectors):
                self.cursor.reset_mock()
                self.embedder.embed_documents.return_value = vectors
                with self.assertRaisesRegex(ValueError, "vector"):
                    self.process([self.chunk()])
                self.assert_no_publication()

    def test_malformed_embeddings_are_rejected(self):
        cases = (
            [[0.1]],
            [[0.1, 0.2, 0.3]],
            [0.1],
            [[[0.1, 0.2]]],
            [["not-a-number", "0.2"]],
            [[True, False]],
            [[1j, 2j]],
            [[float("nan"), 0.2]],
            [[float("inf"), 0.2]],
            [[float("-inf"), 0.2]],
            [[1e100, 0.2]],
            [[0.0, 0.0]],
            [[1e-100, 0.0]],
        )
        for vectors in cases:
            with self.subTest(vectors=vectors):
                self.cursor.reset_mock()
                self.embedder.embed_documents.return_value = vectors
                with self.assertRaisesRegex(ValueError, "[Ee]mbedding"):
                    self.process([self.chunk()])
                self.assert_no_publication()

    def test_invalid_later_vector_prevents_all_publication(self):
        self.embedder.embed_documents.return_value = [[0.1, 0.2], [float("nan"), 0.4]]
        with self.assertRaisesRegex(ValueError, "finite"):
            self.process([self.chunk(), self.chunk(index=1, content="Second text")])
        self.assert_no_publication()

    def test_ragged_embeddings_are_rejected(self):
        self.embedder.embed_documents.return_value = [[0.1, 0.2], [0.3]]
        with self.assertRaisesRegex(ValueError, "[Ee]mbedding"):
            self.process([self.chunk(), self.chunk(index=1, content="Second text")])
        self.assert_no_publication()

    def test_provider_failure_preserves_existing_document(self):
        self.embedder.embed_documents.side_effect = RuntimeError("provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.process([self.chunk()])
        self.assert_no_publication()

    def test_invalid_response_does_not_create_new_document(self):
        self.cursor.fetchone.return_value = None
        self.embedder.embed_documents.return_value = []
        with self.assertRaisesRegex(ValueError, "Expected 1 embedding"):
            self.process([self.chunk()])
        self.assert_no_publication()

    def test_valid_chunks_publish_only_after_embedding(self):
        def embed(texts):
            self.assert_no_publication()
            self.assertEqual(texts, ["Valid text", "Second chunk"])
            return [[0.1, 0.2], [0.3, 0.4]]

        self.embedder.embed_documents.side_effect = embed
        result = self.process([
            self.chunk(), self.chunk(index=1, content="Second chunk", token_count=700)
        ])
        self.assertEqual(result, ("indexed", 2))
        calls = self.conn.execute.call_args_list
        self.assertTrue(calls[0].args[0].startswith("UPDATE documents"))
        self.assertTrue(calls[1].args[0].startswith("DELETE FROM chunks"))
        self.assertEqual(len(calls), 2)
        self.cursor.executemany.assert_called_once()
        query, rows = self.cursor.executemany.call_args.args
        self.assertTrue(query.startswith("INSERT INTO chunks"))
        for index, row in enumerate(rows):
            self.assertEqual(row[0], "existing-document")
            self.assertEqual(row[1], index)
            vector = row[6]
            self.assertEqual(vector.dtype, np.dtype("float32"))
            np.testing.assert_allclose(vector, [[0.1, 0.2], [0.3, 0.4]][index])

    def test_valid_new_document_is_created_after_validation(self):
        self.cursor.fetchone.side_effect = [None, None, {"id": "new-document"}]
        self.assertEqual(self.process([self.chunk()]), ("indexed", 1))
        self.assertTrue(self.cursor.execute.call_args.args[0].startswith("INSERT INTO documents"))
        self.assertEqual(self.conn.execute.call_args.args[1][0], "new-document")


if __name__ == "__main__":
    unittest.main()