import unittest
from unittest.mock import patch

from rag_mcp_platform.ingest.chunker import _markdown_parser, chunk_text


class ChunkLimitTests(unittest.TestCase):
    def chunks(self, text, limit, overlap=0):
        with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
            chunks = chunk_text(text, limit, overlap)
        self.assertTrue(all(0 < chunk.token_count <= limit for chunk in chunks))
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))
        return chunks

    def test_long_unbroken_prose_is_bounded_without_losing_characters(self):
        text = "abcdefghij" * 20
        chunks = self.chunks(text, 30)
        self.assertEqual("".join(chunk.content for chunk in chunks), text)

    def test_multibyte_text_is_not_corrupted_by_hard_splitting(self):
        text = "\u4e2d\u6587\u6d4b\u8bd5" * 20
        chunks = chunk_text(text, 12, 0)
        self.assertTrue(all(chunk.token_count <= 12 for chunk in chunks))
        self.assertEqual("".join(chunk.content for chunk in chunks), text)

    def test_long_whitespace_runs_do_not_create_empty_chunks(self):
        text = "First" + " " * 100 + "Second"
        chunks = self.chunks(text, 20)
        self.assertTrue(all(chunk.content.strip() for chunk in chunks))
        self.assertEqual("".join("".join(chunk.content.split()) for chunk in chunks), "FirstSecond")

    def test_table_splits_between_rows_with_repeated_header(self):
        table = "| Field | Value |\n| --- | --- |\n" + "".join(f"| Item{index} | {index} |\n" for index in range(8))
        chunks = self.chunks(table, 70)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.content.startswith("| Field | Value |\n| --- | --- |"))
            self.assertTrue(any(token.type == "table_open" for token in _markdown_parser().parse(chunk.content)))
        for index in range(8):
            self.assertEqual(sum(f"| Item{index} | {index} |" in chunk.content for chunk in chunks), 1)

    def test_fenced_code_splits_at_lines_with_fences_retained(self):
        lines = [f"    operation_{index}()" for index in range(8)]
        chunks = self.chunks("```python\n" + "\n".join(lines) + "\n```", 70)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            tokens = _markdown_parser().parse(chunk.content)
            self.assertEqual(tokens[0].type, "fence")
            self.assertEqual(tokens[0].info, "python")
            self.assertTrue(chunk.content.endswith("```"))
        for line in lines:
            self.assertEqual(sum(line in chunk.content for chunk in chunks), 1)

    def test_list_splits_between_items(self):
        text = "\n".join(f"- Item {index} with details" for index in range(8))
        chunks = self.chunks(text, 55)
        self.assertGreater(len(chunks), 1)
        for index in range(8):
            self.assertEqual(sum(f"Item {index} with details" in chunk.content for chunk in chunks), 1)

    def test_unsplittable_structured_units_are_rejected(self):
        for text in (
            "```\n" + "x" * 100 + "\n```",
            "| Name | Value |\n| --- | --- |\n| " + "x" * 100 + " | One |",
            "- " + "x" * 100,
        ):
            with self.subTest(text=text):
                with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
                    with self.assertRaisesRegex(ValueError, "Structured block exceeds"):
                        chunk_text(text, 40, 0)

    def test_overlap_never_pushes_a_chunk_over_limit(self):
        self.chunks("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", 35, 20)

    def test_heading_hierarchy_survives_sibling_and_skipped_levels(self):
        chunks = self.chunks("# Guide\n\n## Setup\n\n### Windows\nText.\n\n## Usage\nMore text.", 100)
        self.assertEqual(chunks[0].metadata["section_path"], ["Guide", "Setup", "Windows"])
        self.assertEqual(chunks[1].metadata["section_path"], ["Guide", "Usage"])

    def test_invalid_limits_fail_before_processing(self):
        for maximum, overlap in ((0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)):
            with self.subTest(maximum=maximum, overlap=overlap):
                with self.assertRaisesRegex(ValueError, "Chunk limits"):
                    chunk_text("Text", maximum, overlap)


if __name__ == "__main__":
    unittest.main()