import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.indexer import _process_file, inspect_sources
from rag_mcp_platform.ingest.loaders import LoadedDoc, LoadedPage
from rag_mcp_platform.ingest.pii import PiiRejectedError, sanitize_document


class PiiSanitizationTests(unittest.TestCase):
    def test_names_emails_addresses_and_identifiers_are_replaced_not_rejected(self):
        document = LoadedDoc("guide.txt", "text", "Guide", (
            "Contact Name: Alice Smith; Email: alice@example.org\n"
            "Address: 123 Main Street\nPhone: +1 650-253-0000\n"
            "SSN: 123-45-6789\nCard: 4111 1111 1111 1111\nUseful instructions remain."
        ))
        sanitized = sanitize_document(document)
        for raw in ("Alice Smith", "alice@example.org", "123 Main Street", "+1 650-253-0000", "123-45-6789", "4111 1111 1111 1111"):
            self.assertNotIn(raw, sanitized.text)
        for placeholder in ("<PERSON>", "<EMAIL_ADDRESS>", "<POSTAL_ADDRESS>", "<PHONE_NUMBER>", "<US_SSN>", "<CREDIT_CARD>"):
            self.assertIn(placeholder, sanitized.text)
        self.assertIn("Contact Name:", sanitized.text)
        self.assertIn("Useful instructions remain.", sanitized.text)
        self.assertIn("Alice Smith", document.text)

    def test_nested_metadata_keys_values_titles_and_urls_are_sanitized(self):
        document = LoadedDoc("guide.txt", "text", "alice@example.org", "Safe text.",
            url="https://example.com/alice@example.org", metadata={
                "author": "Alice Smith", "address": "123 Main Street",
                "nested": {"alice@example.org": ["alice@example.org", "Safe text"]},
                "version": 3,
            })
        sanitized = sanitize_document(document)
        serialized = json.dumps(sanitized.metadata) + sanitized.title
        self.assertNotIn("alice@example.org", serialized)
        self.assertNotIn("Alice Smith", serialized)
        self.assertNotIn("123 Main Street", serialized)
        self.assertEqual(sanitized.metadata["version"], 3)
        self.assertEqual(sanitized.metadata["author"], "<PERSON>")
        self.assertIsNone(sanitized.url)

    def test_sensitive_source_identity_is_stable_and_opaque(self):
        document = LoadedDoc("upload://alice@example.org.txt", "text", "Guide", "Safe text.")
        first = sanitize_document(document)
        second = sanitize_document(document)
        self.assertTrue(first.source_path.startswith("sanitized://"))
        self.assertEqual(first.source_path, second.source_path)
        self.assertNotIn("alice", first.source_path)

    def test_pdf_raw_text_and_page_numbers_are_preserved_safely(self):
        document = LoadedDoc("guide.pdf", "pdf", "Guide", "Page content alice@example.org", pages=[
            LoadedPage(3, "Page content alice@example.org", "Header alice@example.org\nPage content"),
        ])
        sanitized = sanitize_document(document)
        self.assertEqual(sanitized.pages[0].page_number, 3)
        self.assertNotIn("alice@example.org", sanitized.pages[0].text)
        self.assertNotIn("alice@example.org", sanitized.pages[0].raw_text)

    def test_clean_documents_are_unchanged(self):
        document = LoadedDoc("guide.md", "markdown", "Guide", "Technical documentation.", metadata={"version": 3})
        self.assertEqual(sanitize_document(document), document)

    def test_repeated_sanitization_is_idempotent(self):
        document = LoadedDoc("guide.txt", "text", "Guide", "Address: 123 Main Street\nContact Name: Alice Smith")
        first = sanitize_document(document)
        self.assertEqual(sanitize_document(first), first)

    def test_detection_and_anonymization_failures_stop_without_exposing_values(self):
        for target in ("_get_recognizers", "_get_anonymizer"):
            with self.subTest(target=target), patch("rag_mcp_platform.ingest.pii." + target, side_effect=RuntimeError("alice@example.org")):
                with self.assertRaises(PiiRejectedError) as raised:
                    sanitize_document(LoadedDoc("guide.txt", "text", "Guide", "alice@example.org"))
                self.assertNotIn("alice@example.org", str(raised.exception))

    def test_ingestion_embeds_and_persists_only_sanitized_representations(self):
        document = LoadedDoc("upload://alice@example.org.txt", "text", "Contact Name: Alice Smith",
            "Contact Name: Alice Smith\nEmail: alice@example.org\nUseful instructions remain.",
            metadata={"author": "Alice Smith"}, url="https://example.com/alice@example.org")
        pool = MagicMock()
        conn = pool.connection.return_value.__enter__.return_value
        conn.cursor.return_value.fetchone.side_effect = [None, None, {"id": "document"}]
        embedder = MagicMock()
        embedder.name = "test"
        embedder.embed_documents.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
        settings = SimpleNamespace(
            processing_fingerprint="pipeline:10", ingestion_pipeline_version="10",
            chunk_max_tokens=700, chunk_overlap_tokens=100, embedding_dim=2,
        )
        info = {}
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space"),
        ):
            result = _process_file(pool, "alice@example.org.txt", settings, embedder, "test:2", True, result_info=info)
        self.assertEqual(result[0], "indexed")
        requests = json.dumps(embedder.embed_documents.call_args.args[0])
        operations = str(conn.execute.call_args_list) + str(conn.cursor.return_value.execute.call_args_list) + str(conn.cursor.return_value.executemany.call_args_list)
        for value in ("Alice Smith", "alice@example.org"):
            self.assertNotIn(value, requests)
            self.assertNotIn(value, operations)
            self.assertNotIn(value, json.dumps(info))
        self.assertIn("Useful instructions remain.", requests)
        self.assertTrue(info["source_path"].startswith("sanitized://"))

    def test_duplicate_warning_never_echoes_original_sensitive_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("alice@example.org.txt", "bob@example.org.txt"):
                (Path(directory) / name).write_text("Contact alice@example.org", encoding="utf-8")
            report = inspect_sources(directory, preview=True)
        self.assertEqual(report["status"], "success")
        self.assertNotIn("alice@example.org", json.dumps(report))
        self.assertNotIn("bob@example.org", json.dumps(report))

    def test_sensitive_filename_does_not_leak_with_a_clean_logical_identity(self):
        document = LoadedDoc("temporary/path.txt", "text", "Guide", "Useful instructions.")
        pool = MagicMock()
        conn = pool.connection.return_value.__enter__.return_value
        conn.cursor.return_value.fetchone.side_effect = [None, None, {"id": "document"}]
        embedder = SimpleNamespace(name="test", embed_documents=lambda texts: [[0.1, 0.2] for _ in texts])
        settings = SimpleNamespace(
            processing_fingerprint="pipeline:10", ingestion_pipeline_version="10",
            chunk_max_tokens=700, chunk_overlap_tokens=100, embedding_dim=2,
        )
        info = {}
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space"),
        ):
            _process_file(pool, "alice@example.org.txt", settings, embedder, "test:2", True,
                          source_path="upload://guide.txt", result_info=info)
        self.assertEqual(info["source_path"], "upload://guide.txt")
        self.assertEqual(info["file"], "[sanitized source]")
        self.assertNotIn("alice@example.org", json.dumps(info))

    def test_anonymizer_returning_unchanged_pii_fails_closed(self):
        anonymizer = MagicMock()
        anonymizer.anonymize.side_effect = lambda **kwargs: SimpleNamespace(text=kwargs["text"], items=[])
        with patch("rag_mcp_platform.ingest.pii._get_anonymizer", return_value=anonymizer):
            with self.assertRaises(PiiRejectedError):
                sanitize_document(LoadedDoc("guide.txt", "text", "Guide", "alice@example.org"))

    def test_sanitization_loads_no_models(self):
        with (
            patch("presidio_analyzer.AnalyzerEngine", side_effect=AssertionError("analyzer constructed")) as analyzer,
            patch("spacy.load", side_effect=AssertionError("model loaded")) as load,
            patch("spacy.cli.download", side_effect=AssertionError("model downloaded")) as download,
        ):
            sanitized = sanitize_document(LoadedDoc("guide.txt", "text", "Guide", "Contact Name: Alice Smith"))
        self.assertIn("<PERSON>", sanitized.text)
        analyzer.assert_not_called()
        load.assert_not_called()
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()