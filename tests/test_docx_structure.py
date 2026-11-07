import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docx import Document
from docx.oxml import OxmlElement

from rag_mcp_platform.ingest.chunker import _markdown_parser, chunk_text
from rag_mcp_platform.ingest.loaders import load_file


class DocxStructureTests(unittest.TestCase):
    def load(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.docx"
            document.save(path)
            return load_file(path)

    def table(self, document, rows):
        table = document.add_table(rows=len(rows), cols=len(rows[0]))
        for row, values in zip(table.rows, rows):
            for cell, value in zip(row.cells, values):
                cell.text = value
        return table

    def test_interleaved_paragraphs_and_tables_keep_document_order(self):
        document = Document()
        document.add_heading("Guide", level=1)
        document.add_paragraph("Before first table.")
        self.table(document, [["Field", "Meaning"], ["Width", "Horizontal size"]])
        document.add_paragraph("Between tables.")
        self.table(document, [["Setting", "Value"], ["Enabled", "Yes"]])
        document.add_paragraph("After second table.")
        loaded = self.load(document)
        expected = [
            "Before first table.", "| Width | Horizontal size |", "Between tables.",
            "| Enabled | Yes |", "After second table.",
        ]
        positions = [loaded.text.index(text) for text in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(loaded.title, "Guide")
        self.assertEqual(loaded.doc_type, "docx")

    def test_table_only_document_is_indexable(self):
        document = Document()
        self.table(document, [["First", "One"], ["Second", "Two"]])
        loaded = self.load(document)
        self.assertIsNotNone(loaded)
        self.assertIn("| First | One |", loaded.text)
        self.assertIn("| Second | Two |", loaded.text)

    def test_title_and_heading_levels_are_preserved(self):
        document = Document()
        document.add_heading("Document title", level=0)
        document.add_heading("Section", level=2)
        document.add_paragraph("Section content.")
        document.add_heading("Details", level=3)
        document.add_paragraph("Detailed content.")
        loaded = self.load(document)
        self.assertIn("# Document title", loaded.text)
        self.assertIn("## Section", loaded.text)
        self.assertIn("### Details", loaded.text)
        with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
            chunks = chunk_text(loaded.text, 700, 100)
        self.assertEqual([chunk.heading for chunk in chunks], ["Section", "Details"])

    def test_cell_paragraphs_preserve_text_and_order(self):
        document = Document()
        table = self.table(document, [["Field", "Description"], ["Notes", "First paragraph."]])
        table.cell(1, 1).add_paragraph("Second paragraph.")
        loaded = self.load(document)
        self.assertIn("First paragraph.", loaded.text)
        self.assertIn("Second paragraph.", loaded.text)
        self.assertLess(loaded.text.index("First paragraph."), loaded.text.index("Second paragraph."))
        tokens = _markdown_parser().parse(loaded.text)
        self.assertEqual(sum(token.type == "td_open" for token in tokens), 4)

    def test_empty_cells_preserve_column_alignment(self):
        document = Document()
        self.table(document, [["Left", "Middle", "Right"], ["", "Value", ""]])
        loaded = self.load(document)
        self.assertIn("|  | Value |  |", loaded.text)

    def test_literal_pipes_do_not_create_extra_columns(self):
        document = Document()
        self.table(document, [["Pattern", "Meaning"], ["A | B", "Either value"]])
        loaded = self.load(document)
        tokens = _markdown_parser().parse(loaded.text)
        self.assertEqual(sum(token.type == "td_open" for token in tokens), 4)
        self.assertIn("A", loaded.text)
        self.assertIn("B", loaded.text)

    def test_horizontal_and_vertical_merges_retain_content(self):
        document = Document()
        table = self.table(document, [["Merged", ""], ["Vertical", "One"], ["", "Two"]])
        table.cell(0, 0).merge(table.cell(0, 1))
        table.cell(1, 0).merge(table.cell(2, 0))
        loaded = self.load(document)
        self.assertIn("Merged", loaded.text)
        self.assertIn("Vertical", loaded.text)
        self.assertIn("One", loaded.text)
        self.assertIn("Two", loaded.text)
        tokens = _markdown_parser().parse(loaded.text)
        self.assertEqual(sum(token.type == "td_open" for token in tokens), 6)

    def test_omitted_grid_column_is_padded(self):
        document = Document()
        table = self.table(document, [["Left", "Right"], ["unused", "Value"]])
        row = table.rows[1]
        row._tr.remove(row.cells[0]._tc)
        before = OxmlElement("w:gridBefore")
        before.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "1")
        row._tr.get_or_add_trPr().append(before)
        loaded = self.load(document)
        self.assertIn("|  | Value |", loaded.text)

    def test_table_remains_attached_to_section_through_chunking(self):
        document = Document()
        document.add_heading("Measurements", level=1)
        document.add_paragraph("Available fields.")
        self.table(document, [["Field", "Units"], ["Width", "Centimeters"]])
        loaded = self.load(document)
        with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
            chunks = chunk_text(loaded.text, 700, 100)
        table_chunk = next(chunk for chunk in chunks if "| Width | Centimeters |" in chunk.content)
        self.assertEqual(table_chunk.heading, "Measurements")

    def test_nested_tables_are_flattened_in_their_original_cell_position(self):
        document = Document()
        table = document.add_table(rows=1, cols=1)
        cell = table.cell(0, 0)
        cell.text = "Before nested table."
        nested = cell.add_table(rows=1, cols=2)
        nested.cell(0, 0).text = "Important nested content"
        nested.cell(0, 1).text = "Second nested value"
        cell.add_paragraph("After nested table.")
        loaded = self.load(document)
        self.assertIn("Nested table row 1, column 1:", loaded.text)
        self.assertIn("Nested table row 1, column 2:", loaded.text)
        positions = [loaded.text.index(text) for text in (
            "Before nested table.", "Important nested content", "Second nested value", "After nested table."
        )]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(sum(token.type == "table_open" for token in _markdown_parser().parse(loaded.text)), 1)

    def test_deeply_nested_table_content_is_not_lost(self):
        document = Document()
        outer = document.add_table(rows=1, cols=1)
        middle = outer.cell(0, 0).add_table(rows=1, cols=1)
        inner = middle.cell(0, 0).add_table(rows=1, cols=1)
        inner.cell(0, 0).text = "Deeply nested value"
        self.assertIn("Deeply nested value", self.load(document).text)

    def test_empty_document_and_empty_table_are_not_indexable(self):
        self.assertIsNone(self.load(Document()))
        document = Document()
        document.add_table(rows=2, cols=2)
        self.assertIsNone(self.load(document))


if __name__ == "__main__":
    unittest.main()