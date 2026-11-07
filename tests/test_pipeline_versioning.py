import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.indexer import _document_content_hash, _process_file
from rag_mcp_platform.ingest.loaders import LoadedDoc
from rag_mcp_platform.settings import Settings


class PipelineVersioningTests(unittest.TestCase):
    def setUp(self):
        guard = patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space")
        guard.start()
        self.addCleanup(guard.stop)

    def settings(self, **overrides):
        configuration = {
            "ingestion_pipeline_version": "1",
            "chunk_max_tokens": 700,
            "chunk_overlap_tokens": 100,
            "embedding_dim": 2,
        }
        configuration.update(overrides)
        return Settings(_env_file=None, **configuration)

    def process(self, settings, existing, *, full=False, duplicate=None):
        pool = MagicMock()
        conn = pool.connection.return_value.__enter__.return_value
        cursor = conn.cursor.return_value
        cursor.fetchone.side_effect = (
            [existing, existing] if existing
            else [None, duplicate, None, duplicate, {"id": "new-document"}]
        )
        embedder = MagicMock()
        embedder.name = "test"
        embedder.embed_documents.return_value = [[0.1, 0.2]]
        document = LoadedDoc("guide.txt", "text", "Guide", "unchanged content")
        document.metadata["author"] = "Documentation team"
        chunk = SimpleNamespace(index=0, heading=None, content=document.text, token_count=2, metadata={})
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer.chunk_text", return_value=[chunk]),
        ):
            result = _process_file(pool, "guide.txt", settings, embedder, "test:2", full)
        return result, conn, cursor, embedder

    def existing(self, fingerprint):
        return {
            "id": "existing-document",
            "version": 1,
            "content_hash": _document_content_hash(LoadedDoc(
                "guide.txt", "text", "Guide", "unchanged content",
                metadata={"author": "Documentation team"},
            )),
            "embedding_signature": "test:2",
            "metadata": {"processing_fingerprint": fingerprint} if fingerprint else {},
        }

    def test_fingerprint_is_deterministic(self):
        self.assertEqual(self.settings().processing_fingerprint, self.settings().processing_fingerprint)

    def test_chunk_settings_and_pipeline_version_change_fingerprint(self):
        original = self.settings().processing_fingerprint
        for overrides in (
            {"chunk_max_tokens": 350},
            {"chunk_overlap_tokens": 50},
            {"ingestion_pipeline_version": "2"},
        ):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(original, self.settings(**overrides).processing_fingerprint)

    def test_retrieval_settings_do_not_change_fingerprint(self):
        self.assertEqual(
            self.settings().processing_fingerprint,
            self.settings(top_k=20, rrf_k=80).processing_fingerprint,
        )

    def test_unchanged_document_skips_when_fingerprint_matches(self):
        settings = self.settings()
        result, conn, _, embedder = self.process(settings, self.existing(settings.processing_fingerprint))
        self.assertEqual(result, ("skipped", 0))
        conn.execute.assert_not_called()
        embedder.embed_documents.assert_not_called()

    def test_stale_and_legacy_documents_are_reprocessed(self):
        for fingerprint in (None, "old-fingerprint"):
            with self.subTest(fingerprint=fingerprint):
                settings = self.settings()
                result, conn, _, embedder = self.process(settings, self.existing(fingerprint))
                self.assertEqual(result, ("indexed", 1))
                embedder.embed_documents.assert_called_once()
                metadata = json.loads(conn.execute.call_args_list[0].args[1][-2])
                self.assertEqual(metadata["processing_fingerprint"], settings.processing_fingerprint)
                self.assertEqual(metadata["ingestion_pipeline_version"], "1")
                self.assertEqual(metadata["author"], "Documentation team")

    def test_deduplication_is_scoped_to_processing_fingerprint(self):
        settings = self.settings()
        result, _, cursor, embedder = self.process(
            settings, None, duplicate={"source_path": "another.txt"}
        )
        self.assertEqual(result, ("duplicate", 0))
        query = cursor.execute.call_args_list[1]
        self.assertIn("metadata->>'processing_fingerprint' = %s", query.args[0])
        self.assertEqual(query.args[1][-1], settings.processing_fingerprint)
        embedder.embed_documents.assert_not_called()

    def test_full_reindex_bypasses_matching_fingerprint(self):
        settings = self.settings()
        result, _, _, embedder = self.process(
            settings, self.existing(settings.processing_fingerprint), full=True
        )
        self.assertEqual(result, ("indexed", 1))
        embedder.embed_documents.assert_called_once()


if __name__ == "__main__":
    unittest.main()