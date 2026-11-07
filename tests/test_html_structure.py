import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_mcp_platform.ingest.chunker import _split_sections, chunk_text
from rag_mcp_platform.ingest.loaders import load_file
from rag_mcp_platform.settings import Settings


class HtmlStructureTests(unittest.TestCase):
    def load(self, html):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.htm"
            path.write_bytes(html.encode("utf-8") if isinstance(html, str) else html)
            return load_file(path)

    def test_main_content_headings_and_inline_text_survive(self):
        document = self.load('''<html><head><title>Guide title</title></head><body>
            <nav>Navigation noise</nav><div>Outside content</div>
            <div id="mc-main-content"><h1>Overview</h1><p>First <strong>important</strong> step.</p>
            <h2>Details</h2><p>Use <code>item_count</code> here.</p></div></body></html>''')
        self.assertEqual(document.title, "Guide title")
        self.assertIn("# Overview", document.text)
        self.assertIn("## Details", document.text)
        self.assertIn("First **important** step.", document.text)
        self.assertIn("`item_count`", document.text)
        self.assertNotIn("Navigation noise", document.text)
        self.assertNotIn("Outside content", document.text)

    def test_table_cells_and_header_relationships_survive(self):
        document = self.load('''<h1>Fields</h1><table><thead><tr><th>Field</th><th>Description</th>
            </tr></thead><tbody><tr><td>Width</td><td>Horizontal size</td></tr>
            <tr><td>Height</td><td>Vertical size</td></tr></tbody></table>''')
        self.assertIn("| Field | Description |", document.text)
        self.assertIn("| Width | Horizontal size |", document.text)
        self.assertIn("| Height | Vertical size |", document.text)

    def test_table_without_headers_does_not_discard_first_row(self):
        document = self.load("<table><tr><td>First</td><td>One</td></tr>"
                             "<tr><td>Second</td><td>Two</td></tr></table>")
        self.assertIn("| First | One |", document.text)
        self.assertIn("| Second | Two |", document.text)

    def test_code_indentation_blank_lines_and_heading_characters_survive_chunking(self):
        document = self.load("<main><h1>Example</h1><pre><code>if ready:\n"
                             "    run()\n\n# not a heading\n    next_step()\n</code></pre></main>")
        self.assertIn("```", document.text)
        code = "if ready:\n    run()\n\n# not a heading\n    next_step()"
        self.assertIn(code, document.text)
        with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
            chunks = chunk_text(document.text, 700, 100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading, "Example")
        self.assertIn(code, chunks[0].content)

    def test_courierline_runs_become_separate_code_blocks(self):
        document = self.load('''<main><h1>Automation</h1>
            <p class="courierline">if (ready) {</p>
            <p class="courierline">&#160;&#160;&#160;&#160;run();</p>
            <p class="courierline">&#160;</p>
            <p class="courierline">}</p>
            <p>Explanation.</p><p class="courierline">next();</p></main>''')
        self.assertIn("if (ready) {\n    run();\n \n}", document.text)
        self.assertEqual(document.text.count("```"), 4)
        self.assertLess(document.text.index("Explanation."), document.text.index("next();"))

    def test_generic_body_removes_toolbars_and_navigation(self):
        document = self.load('''<body><header>Site</header><nav>Menu</nav>
            <div role="Navigation">Sidebar</div><div class="nocontent">Print help</div>
            <h1>Topic</h1><p>Useful content.</p><script>bad()</script>
            <style>bad{}</style><footer>Copyright</footer></body>''')
        self.assertIn("Useful content.", document.text)
        for noise in ("Menu", "Sidebar", "Print help", "bad()", "bad{}", "Copyright"):
            self.assertNotIn(noise, document.text)

    def test_declared_encoding_is_respected(self):
        document = self.load(b'<html><head><meta charset="windows-1252"></head>'
                             b'<body><h1>Caf\xe9</h1><p>Text.</p></body></html>')
        self.assertIn("Caf\u00e9", document.text)

    def test_empty_page_is_not_indexable(self):
        self.assertIsNone(self.load("<html><head><title>Title</title></head><body></body></html>"))


class StructuredChunkTests(unittest.TestCase):
    def test_fenced_heading_is_not_a_section(self):
        sections = _split_sections("# Example\n\n```python\n# comment\nrun()\n```\n\n## Next\nText")
        self.assertEqual([heading for heading, _ in sections], ["Example", "Next"])

    def test_indented_code_whitespace_is_preserved(self):
        with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
            chunks = chunk_text("# Example\n\n    run()\n\n    next()\n", 100, 10)
        self.assertEqual(chunks[0].content, "    run()\n\n    next()")

    def test_code_and_table_blocks_are_not_split_between_chunks(self):
        for block in ("```\none()\n\n    two()\n```", "| A | B |\n| --- | --- |\n| 1 | 2 |"):
            with self.subTest(block=block):
                with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
                    chunks = chunk_text(f"# Topic\n\nIntro text.\n\n{block}\n\nEnding text.", 40, 0)
                self.assertTrue(any(block in chunk.content for chunk in chunks))
                self.assertTrue(all(chunk.token_count <= 40 for chunk in chunks))

    def test_oversized_structured_blocks_are_rejected(self):
        with patch("rag_mcp_platform.ingest.chunker._token_len", return_value=len):
            with self.assertRaisesRegex(ValueError, "Structured block exceeds"):
                chunk_text("# Topic\n\n```\n" + "code " * 20 + "\n```", 40, 0)

    def test_setext_headings_are_recognized(self):
        self.assertEqual(_split_sections("Title\n=====\n\nBody"), [("Title", "Body")])

    def test_default_pipeline_version_is_bumped(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings(_env_file=None).ingestion_pipeline_version, "10")


if __name__ == "__main__":
    unittest.main()