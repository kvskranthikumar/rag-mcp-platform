import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_mcp_platform.ingest.evaluation import evaluate_manifest


class IngestionEvaluationTests(unittest.TestCase):
    def evaluate(self, cases, *, content="# Guide\n\n## Setup\n\nInstall the package."):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text(content, encoding="utf-8")
            manifest = root / "evaluation.json"
            manifest.write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")
            with (
                patch("rag_mcp_platform.ingest.indexer.get_pool", side_effect=AssertionError("database accessed")),
                patch("rag_mcp_platform.ingest.indexer.get_embedding_provider", side_effect=AssertionError("provider accessed")),
            ):
                return evaluate_manifest(str(manifest))

    def test_quality_checks_pass_for_expected_content_and_hierarchy(self):
        report = self.evaluate([{
            "path": "guide.md", "expected_text": ["Install the package."], "forbidden_text": ["Navigation noise"],
            "expected_section_paths": [["Guide", "Setup"]], "max_chunk_tokens": 700, "min_chunks": 1,
        }])
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["status"], "success")

    def test_failed_checks_are_identified_without_echoing_content(self):
        report = self.evaluate([{"path": "guide.md", "expected_text": ["missing private fragment"]}])
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["results"][0]["failed_checks"], ["expected_text[0]"])
        self.assertNotIn("missing private fragment", json.dumps(report))

    def test_pii_replacement_can_be_evaluated(self):
        report = self.evaluate([{
            "path": "guide.md", "expected_status": "ready",
            "expected_text": ["<EMAIL_ADDRESS>"], "forbidden_text": ["alice@example.org"],
        }], content="Contact alice@example.org")
        self.assertEqual(report["status"], "success")
        self.assertNotIn("alice@example.org", json.dumps(report))

    def test_missing_source_is_a_safe_case_failure(self):
        report = self.evaluate([{"path": "missing.md"}])
        self.assertEqual(report["results"][0]["failed_checks"], ["inspection_error"])

    def test_invalid_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            for manifest in ({"version": 2, "cases": []}, {"version": 1, "cases": []}):
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate_manifest(str(path))


if __name__ == "__main__":
    unittest.main()