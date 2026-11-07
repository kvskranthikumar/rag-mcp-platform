import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.indexer import _process_file, ingest_files, inspect_sources
from rag_mcp_platform.ingest.loaders import LoadedDoc, LoadedPage
from rag_mcp_platform.ingest.pii import PiiRejectedError, _get_recognizers, reject_pii


class PiiRejectionTests(unittest.TestCase):
    def document(self, text="Safe technical documentation.", **options):
        return LoadedDoc("guide.txt", "text", "Guide", text, **options)

    def test_supported_structured_identifiers_are_rejected_without_values_in_errors(self):
        for value in (
            "alice@example.org", "+1 650-253-0000", "123-45-6789", "4111 1111 1111 1111",
            "4111-1111-1111-1111", "4111111111111111",
        ):
            with self.subTest(value=value):
                with self.assertRaises(PiiRejectedError) as raised:
                    reject_pii(self.document("Contact or identifier: " + value))
                self.assertIn("PII policy rejected", str(raised.exception))
                self.assertNotIn(value, str(raised.exception))

    def test_metadata_url_and_source_labels_are_screened(self):
        email = "alice@example.org"
        documents = [
            self.document(metadata={"author": email}),
            self.document(url=f"https://example.com/{email}"),
            LoadedDoc(email + ".txt", "text", "Guide", "Safe text."),
            LoadedDoc("guide.txt", "text", email, "Safe text."),
        ]
        for document in documents:
            with self.subTest(document=document):
                with self.assertRaises(PiiRejectedError):
                    reject_pii(document)

    def test_isolated_pdf_page_text_is_screened(self):
        with self.assertRaises(PiiRejectedError):
            reject_pii(self.document(pages=[LoadedPage(1, "alice@example.org")]))

    def test_versions_formulas_and_code_are_not_rejected_as_identifiers(self):
        reject_pii(self.document(
            "Version 2025.4.0.0. Width=120.5; Height=30; UPC=012345678905. "
            "public bool Activate(int index) { return index > 0; }"
        ))

    def test_labeled_english_person_names_are_rejected(self):
        for value in ("Contact Name: Alice Smith", "Customer: Mary Jane Watson", "Prepared by: John O'Neil"):
            with self.subTest(value=value):
                with self.assertRaises(PiiRejectedError) as raised:
                    reject_pii(self.document(value))
                self.assertIn("person_name=1", str(raised.exception))
                self.assertNotIn(value.partition(": ")[2], str(raised.exception))

    def test_labeled_and_structured_postal_addresses_are_rejected(self):
        for value in (
            "Address: 123 Main Street, Springfield, IL 62704",
            "Mailing Address: 44 North Pine Ave",
            "Address: P.O. Box 123",
            "Send to 123 Main Street, Springfield IL 62704",
        ):
            with self.subTest(value=value):
                with self.assertRaises(PiiRejectedError) as raised:
                    reject_pii(self.document(value))
                self.assertIn("postal_address=1", str(raised.exception))
                self.assertNotIn(value, str(raised.exception))

    def test_technical_labels_values_and_versions_do_not_trigger_name_rules(self):
        reject_pii(self.document(
            "Product Name: Example Platform. Project: Example Organization. "
            "Field Name: Customer. Version: 2025.4.0.0. "
            "Use 123 Main Street as a fictional example."
        ))

    def test_pii_categories_are_counted_without_echoing_values(self):
        with self.assertRaises(PiiRejectedError) as raised:
            reject_pii(self.document("Customer: Alice Smith; Address: 123 Main Street"))
        self.assertIn("person_name=1", str(raised.exception))
        self.assertIn("postal_address=1", str(raised.exception))
        self.assertNotIn("Alice Smith", str(raised.exception))
        self.assertNotIn("123 Main Street", str(raised.exception))

    def test_detector_failure_rejects_without_exposing_exception_text(self):
        with patch("rag_mcp_platform.ingest.pii._get_recognizers", side_effect=RuntimeError("alice@example.org")):
            with self.assertRaisesRegex(PiiRejectedError, "screening unavailable") as raised:
                reject_pii(self.document())
        self.assertNotIn("alice@example.org", str(raised.exception))

    def test_presidio_does_not_construct_analyzer_or_load_nlp_models(self):
        _get_recognizers.cache_clear()
        with (
            patch("presidio_analyzer.AnalyzerEngine", side_effect=AssertionError("analyzer constructed")) as analyzer,
            patch("spacy.load", side_effect=AssertionError("NLP model loaded")) as load,
            patch("spacy.cli.download", side_effect=AssertionError("NLP model downloaded")) as download,
        ):
            reject_pii(self.document("Technical documentation."))
            with self.assertRaises(PiiRejectedError):
                reject_pii(self.document("Contact Name: Alice Smith"))
        analyzer.assert_not_called()
        load.assert_not_called()
        download.assert_not_called()

    def test_invalid_card_checksum_is_not_rejected_as_a_card(self):
        reject_pii(self.document("Card-shaped test value: 4111 1111 1111 1112"))

    def test_overlapping_presidio_matches_are_counted_once(self):
        with self.assertRaises(PiiRejectedError) as raised:
            reject_pii(self.document("Contact alice@example.org"))
        self.assertIn("email=1", str(raised.exception))

    def test_sanitization_failure_precedes_database_deduplication_and_embedding_even_with_full(self):
        for full in (False, True):
            with self.subTest(full=full):
                pool = MagicMock()
                embedder = MagicMock()
                with (
                    patch("rag_mcp_platform.ingest.indexer.load_file", return_value=self.document("alice@example.org")),
                    patch("rag_mcp_platform.ingest.indexer.sanitize_document", side_effect=PiiRejectedError("Sanitization failed.")),
                ):
                    with self.assertRaises(PiiRejectedError):
                        _process_file(pool, "guide.txt", SimpleNamespace(), embedder, "test:2", full)
                pool.connection.assert_not_called()
                embedder.embed_documents.assert_not_called()

    def test_inspection_previews_sanitized_content_without_sensitive_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alice@example.org.txt"
            path.write_text("Contact alice@example.org", encoding="utf-8")
            report = inspect_sources(str(path), preview=True)
        serialized = json.dumps(report)
        self.assertEqual(report["status"], "success")
        self.assertNotIn("alice@example.org", serialized)
        self.assertIn("chunks", report["results"][0])
        self.assertTrue(report["results"][0]["source_path"].startswith("sanitized://"))

    def test_batch_continues_after_pii_rejection_and_records_safe_error(self):
        settings = SimpleNamespace(embedding_dim=2, embedding_signature="test:2")
        with (
            patch("rag_mcp_platform.ingest.indexer.get_settings", return_value=settings),
            patch("rag_mcp_platform.ingest.indexer.get_embedding_provider", return_value=SimpleNamespace(dimension=2)),
            patch("rag_mcp_platform.ingest.indexer.get_pool"),
            patch("rag_mcp_platform.ingest.indexer._start_run", return_value=1),
            patch("rag_mcp_platform.ingest.indexer._finish_run"),
            patch("rag_mcp_platform.ingest.indexer._record_file_result") as record,
            patch("rag_mcp_platform.ingest.indexer._process_file", side_effect=[
                PiiRejectedError("PII policy rejected document (email=1)."), ("indexed", 1),
            ]),
        ):
            report = ingest_files(["alice@example.org.txt", "safe.txt"])
        self.assertEqual(report["status"], "partial_success")
        self.assertNotIn("alice@example.org", json.dumps(report))
        self.assertEqual(record.call_args_list[0].args[2]["file"], "[PII rejected]")


if __name__ == "__main__":
    unittest.main()