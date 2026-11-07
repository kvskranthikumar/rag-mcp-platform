import json
import tempfile
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf

from rag_mcp_platform.ingest.indexer import _content_hash, _document_content_hash, _process_file
from rag_mcp_platform.ingest.loaders import LoadedDoc, LoadedPage, load_file
from rag_mcp_platform.models import RetrievedChunk
from rag_mcp_platform.ingest.pii import PiiRejectedError, reject_pii
from rag_mcp_platform.settings import Settings


class PdfProvenanceTests(unittest.TestCase):
    def load(self, texts, *, metadata=None, password=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.pdf"
            with pymupdf.open() as document:
                for text in texts:
                    page = document.new_page()
                    if text:
                        page.insert_text((72, 72), text)
                if metadata:
                    document.set_metadata(metadata)
                if password:
                    document.save(
                        path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                        owner_pw=password, user_pw=password,
                    )
                else:
                    document.save(path)
            return load_file(path)

    def index(self, document, *, max_tokens=700):
        pool = MagicMock()
        conn = pool.connection.return_value.__enter__.return_value
        conn.cursor.return_value.fetchone.side_effect = [None, None, {"id": "document"}]
        embedder = MagicMock()
        embedder.name = "test"
        embedder.embed_documents.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
        settings = SimpleNamespace(
            chunk_max_tokens=max_tokens, chunk_overlap_tokens=0, embedding_dim=2,
            processing_fingerprint="pipeline:4", ingestion_pipeline_version="4",
        )
        with (
            patch("rag_mcp_platform.ingest.indexer.load_file", return_value=document),
            patch("rag_mcp_platform.ingest.indexer._require_compatible_embedding_space"),
            patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len),
        ):
            result = _process_file(pool, "guide.pdf", settings, embedder, "test:2", True)
        chunks = [
            call.args[1] for call in conn.execute.call_args_list
            if call.args[0].startswith("INSERT INTO chunks")
        ]
        for call in conn.cursor.return_value.executemany.call_args_list:
            chunks.extend(call.args[1])
        return result, chunks, conn, embedder

    def test_pdf_metadata_and_physical_page_numbers_are_preserved(self):
        document = self.load(
            ["First page.", "", "Third page."],
            metadata={"title": "PDF Guide", "author": "Documentation team"},
        )
        self.assertEqual(document.title, "PDF Guide")
        self.assertEqual(document.doc_type, "pdf")
        self.assertEqual([page.page_number for page in document.pages], [1, 2, 3])
        self.assertEqual(document.metadata["page_count"], 3)
        self.assertEqual(document.metadata["pages_without_text"], [2])
        self.assertEqual(document.metadata["pdf_metadata"]["author"], "Documentation team")
        self.assertEqual(document.metadata["extraction_method"], "pymupdf_text")

    def test_untitled_pdf_uses_extracted_text_as_title(self):
        self.assertEqual(self.load(["Visible title"]).title, "Visible title")

    def test_text_is_sorted_by_position(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordered.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 200), "Lower text")
                page.insert_text((72, 72), "Upper text")
                document.save(path)
            loaded = load_file(path)
        self.assertLess(loaded.pages[0].text.index("Upper text"), loaded.pages[0].text.index("Lower text"))

    def test_pdf_without_extractable_text_reports_ocr_requirement(self):
        with self.assertRaisesRegex(ValueError, "no extractable text.*OCR"):
            self.load([""])

    def test_password_protected_pdf_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Password-protected PDF"):
            self.load(["Protected content."], password="test-only-password")

    def test_chunks_do_not_cross_pages_or_renumber_empty_pages(self):
        document = self.load(["Only first-page text.", "", "Only third-page text."])
        result, chunks, _, embedder = self.index(document)
        self.assertEqual(result, ("indexed", 2))
        self.assertEqual([chunk[1] for chunk in chunks], [0, 1])
        self.assertEqual([json.loads(chunk[8])["page_number"] for chunk in chunks], [1, 3])
        self.assertEqual([chunk[5] for chunk in chunks], ["page=1", "page=3"])
        self.assertNotIn("third-page", chunks[0][3])
        self.assertNotIn("first-page", chunks[1][3])
        self.assertEqual(embedder.embed_documents.call_args.args[0], [chunk[3] for chunk in chunks])

    def test_multiple_chunks_keep_the_same_page_number(self):
        document = self.load(["First short sentence. Second short sentence. Third short sentence."])
        result, chunks, _, _ = self.index(document, max_tokens=30)
        self.assertGreater(result[1], 1)
        self.assertTrue(all(json.loads(chunk[8])["page_number"] == 1 for chunk in chunks))
        self.assertEqual([chunk[1] for chunk in chunks], list(range(len(chunks))))

    def test_document_metadata_is_persisted_with_chunks(self):
        document = self.load(["Page content."], metadata={"title": "Guide", "author": "Team"})
        _, _, conn, _ = self.index(document)
        query = next(
            call for call in conn.cursor.return_value.execute.call_args_list
            if call.args[0].startswith("INSERT INTO documents")
        )
        metadata = json.loads(query.args[1][-1])
        self.assertEqual(metadata["pdf_metadata"]["author"], "Team")
        self.assertEqual(metadata["page_count"], 1)
        self.assertEqual(metadata["processing_fingerprint"], "pipeline:4")

    def test_page_boundary_changes_invalidate_hash_even_when_text_is_identical(self):
        first = LoadedDoc("guide.pdf", "pdf", "Guide", "alpha\n\nbeta", pages=[
            LoadedPage(1, "alpha\n\nbeta"),
        ])
        second = LoadedDoc("guide.pdf", "pdf", "Guide", first.text, pages=[
            LoadedPage(1, "alpha"), LoadedPage(2, "beta"),
        ])
        self.assertNotEqual(_document_content_hash(first), _document_content_hash(second))
        self.assertEqual(_document_content_hash(first), _document_content_hash(first))

    def test_non_pdf_hashes_remain_unchanged(self):
        document = LoadedDoc("guide.md", "markdown", "Guide", "Text")
        self.assertEqual(_document_content_hash(document), _content_hash(document.text))

    def test_pdf_table_is_preserved_in_reading_order_without_duplicate_cell_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "Before table.")
                for horizontal in (110, 140, 170):
                    page.draw_line((72, horizontal), (360, horizontal))
                for vertical in (72, 200, 360):
                    page.draw_line((vertical, 110), (vertical, 170))
                for point, text in (((80, 130), "Field"), ((210, 130), "Value"), ((80, 160), "Width"), ((210, 160), "Centimeters")):
                    page.insert_text(point, text)
                page.insert_text((72, 220), "After table.")
                document.save(path)
            loaded = load_file(path)
        self.assertEqual(loaded.metadata["table_count"], 1)
        self.assertIn("|", loaded.text)
        self.assertEqual(loaded.text.count("Centimeters"), 1)
        self.assertLess(loaded.text.index("Before table."), loaded.text.index("Centimeters"))
        self.assertLess(loaded.text.index("Centimeters"), loaded.text.index("After table."))

    def test_repeated_margin_cleanup_is_opt_in_and_does_not_bypass_pii(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "margins.pdf"
            with pymupdf.open() as document:
                for index in range(3):
                    page = document.new_page()
                    page.insert_text((72, 40), "Contact alice@example.org")
                    page.insert_text((72, 180), f"Unique body {index}.")
                    page.insert_text((72, 810), "Repeated footer")
                document.save(path)
            unchanged = load_file(path)
            self.assertIn("alice@example.org", unchanged.text)
            with patch("rag_mcp_platform.ingest.loaders.get_settings", return_value=Settings(
                _env_file=None, pdf_remove_repeated_margins=True,
            )):
                cleaned = load_file(path)
        self.assertNotIn("alice@example.org", cleaned.text)
        self.assertNotIn("Repeated footer", cleaned.text)
        self.assertEqual(cleaned.metadata["removed_repeated_margin_lines"], 6)
        for index in range(3):
            self.assertIn(f"Unique body {index}.", cleaned.text)
        with self.assertRaises(PiiRejectedError):
            reject_pii(cleaned)

    def test_pdf_inspection_cli_emits_only_json(self):
        from rag_mcp_platform.ingest.cli import main

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.pdf"
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "Technical documentation.")
                document.save(path)
            output = StringIO()
            with (
                patch.object(sys, "argv", ["rag-mcp-platform", "inspect", "--path", str(path)]),
                patch.object(pymupdf, "_recommend_layout", True, create=True),
                redirect_stdout(output),
            ):
                main()
        self.assertEqual(json.loads(output.getvalue())["status"], "success")


class PageCitationTests(unittest.TestCase):
    def chunk(self, metadata, *, url=None, anchor=None):
        return RetrievedChunk(
            chunk_id=1, document_id="document", content="Text", heading="Section",
            url_anchor=anchor, title="Guide", source_path="guide.pdf", url=url,
            metadata=metadata,
        )

    def test_page_number_is_visible_in_citation(self):
        self.assertEqual(self.chunk({"page_number": 3}).citation(), "Guide > Section (page 3)")

    def test_pdf_url_anchor_and_page_label_are_preserved(self):
        self.assertEqual(
            self.chunk({"page_number": 3}, url="https://example.com/guide.pdf", anchor="page=3").citation(),
            "Guide > Section (page 3) (https://example.com/guide.pdf#page=3)",
        )

    def test_non_page_citations_are_unchanged(self):
        for value in (None, 0, -1, True, "3"):
            with self.subTest(value=value):
                self.assertEqual(self.chunk({"page_number": value}).citation(), "Guide > Section")
        self.assertEqual(self.chunk({}).citation(), "Guide > Section")


if __name__ == "__main__":
    unittest.main()