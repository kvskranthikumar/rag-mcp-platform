import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_mcp_platform.ingest.chunker import Chunk
from rag_mcp_platform.ingest.indexer import _embed_prepared_chunks
from rag_mcp_platform.telemetry import capture_metrics


class EmbeddingReuseTests(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock()
        self.cursor = self.pool.connection.return_value.__enter__.return_value.cursor.return_value
        self.settings = SimpleNamespace(embedding_reuse_enabled=True, embedding_dim=2)
        self.embedder = MagicMock()
        self.embedder.embed_documents.side_effect = lambda texts: [[0.3, 0.4] for _ in texts]

    def chunks(self, *texts):
        return [Chunk(index, None, text, 1) for index, text in enumerate(texts)]

    def test_only_missing_chunks_are_embedded_and_order_is_preserved(self):
        self.cursor.fetchall.return_value = [{"content": "cached", "embedding": [0.1, 0.2]}]
        with capture_metrics() as metrics:
            vectors = _embed_prepared_chunks(
                self.pool, self.chunks("cached", "new", "cached"), self.settings, self.embedder, "model:2",
            )
        self.embedder.embed_documents.assert_called_once_with(["new"])
        self.assertEqual(vectors.shape, (3, 2))
        self.assertEqual(metrics["embedding_reuse_hits"], 2)
        self.assertEqual(metrics["embedding_inputs"], 1)
        self.assertEqual(self.cursor.execute.call_args.args[1][1], "model:2")

    def test_invalid_cached_vectors_are_not_reused(self):
        self.cursor.fetchall.return_value = [{"content": "cached", "embedding": [float("nan"), 0.2]}]
        with capture_metrics() as metrics:
            _embed_prepared_chunks(self.pool, self.chunks("cached"), self.settings, self.embedder, "model:2")
        self.embedder.embed_documents.assert_called_once_with(["cached"])
        self.assertEqual(metrics["embedding_reuse_invalid"], 1)

    def test_hash_collision_does_not_reuse_different_content(self):
        self.cursor.fetchall.return_value = [{"content": "different", "embedding": [0.1, 0.2]}]
        _embed_prepared_chunks(self.pool, self.chunks("wanted"), self.settings, self.embedder, "model:2")
        self.embedder.embed_documents.assert_called_once_with(["wanted"])

    def test_fully_cached_document_makes_no_embedding_request(self):
        self.cursor.fetchall.return_value = [{"content": "cached", "embedding": [0.1, 0.2]}]
        _embed_prepared_chunks(self.pool, self.chunks("cached"), self.settings, self.embedder, "model:2")
        self.embedder.embed_documents.assert_not_called()

    def test_disabled_reuse_does_not_access_database(self):
        self.settings.embedding_reuse_enabled = False
        _embed_prepared_chunks(self.pool, self.chunks("same", "same"), self.settings, self.embedder, "model:2")
        self.pool.connection.assert_not_called()
        self.embedder.embed_documents.assert_called_once_with(["same"])


if __name__ == "__main__":
    unittest.main()