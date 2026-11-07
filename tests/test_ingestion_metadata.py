import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.chunker import Chunk
from rag_mcp_platform.ingest.indexer import _document_content_hash, _process_file
from rag_mcp_platform.ingest.loaders import LoadedDoc, load_file


class IngestionMetadataTests(unittest.TestCase):
    def load(self, head, *, language='lang="en-us"'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.htm"
            path.write_text(
                f"<html {language}><head>{head}</head><body><h1>Guide</h1><p>Text.</p></body></html>",
                encoding="utf-8",
            )
            return load_file(path)

    def test_source_metadata_is_extracted_without_inventing_permissions(self):
        document = self.load('''<title>Source title</title><meta name="product" content="Example Platform">
            <meta name="version" content="2025.4"><meta name="author" content="Team">
            <meta name="audience" content="External"><meta name="category" content="Planning">
            <meta name="document" content="Automation Help"><meta name="keywords" content="fixtures, formulas, fixtures">
            <meta name="tenant" content="untrusted-tenant">''')
        self.assertEqual(document.metadata, {
            "source_title": "Source title", "product": "Example Platform", "product_version": "2025.4",
            "author": "Team", "audience": "External", "category": "Planning",
            "document_type": "Automation Help", "tags": ["fixtures", "formulas"], "language": "en-us",
        })

    def test_absolute_canonical_url_is_preserved_without_fetching(self):
        document = self.load('<link rel="canonical" href="https://example.com/guide.htm">')
        self.assertEqual(document.url, "https://example.com/guide.htm")

    def test_invalid_relative_and_credentialed_urls_are_ignored(self):
        for url in ("guide.htm", "javascript:alert(1)", "file:///guide.htm", "https://user:pass@example.com/guide", "https://["):
            with self.subTest(url=url):
                self.assertIsNone(self.load(f'<link rel="canonical" href="{url}">').url)

    def test_missing_metadata_is_not_guessed(self):
        document = self.load("", language="")
        self.assertEqual(document.metadata, {})

    def test_metadata_only_changes_invalidate_hash(self):
        original = self.load('<meta name="version" content="1">')
        changed = self.load('<meta name="version" content="2">')
        self.assertEqual(original.text, changed.text)
        self.assertNotEqual(_document_content_hash(original), _document_content_hash(changed))

    def test_runtime_bookkeeping_does_not_change_hash(self):
        document = LoadedDoc("guide.txt", "text", "Guide", "Text")
        original = _document_content_hash(document)
        document.metadata.update(processing_fingerprint="new", ingestion_pipeline_version="5", original_filename="guide.txt")
        self.assertEqual(_document_content_hash(document), original)

    def test_source_metadata_reaches_chunks_without_overwriting_page_provenance(self):
        document = LoadedDoc("guide.pdf", "pdf", "Guide", "Text", metadata={
            "product": "Example Platform", "product_version": "2025.4", "language": "en-us",
        })
        chunk = Chunk(0, None, "Text", 1, metadata={"page_number": 3})
        pool = MagicMock()
        conn = pool.connection.return_value.__enter__.return_value
        conn.cursor.return_value.fetchone.side_effect = [None, None, {"id": "document"}]
        settings = SimpleNamespace(
            processing_fingerprint="pipeline:5", ingestion_pipeline_version="5",
            chunk_max_tokens=700, chunk_overlap_tokens=100, embedding_dim=2,
        )
        embedder = SimpleNamespace(name="test", embed_documents=lambda texts: [[0.1, 0.2]])
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer.chunk_text", return_value=[chunk]),
            patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space"),
        ):
            self.assertEqual(_process_file(pool, "guide.pdf", settings, embedder, "test:2", True), ("indexed", 1))
        metadata = json.loads(conn.execute.call_args.args[1][-1])
        self.assertEqual(metadata["product_version"], "2025.4")
        self.assertEqual(metadata["source_path"], "guide.pdf")
        self.assertEqual(metadata["page_number"], 3)


if __name__ == "__main__":
    unittest.main()