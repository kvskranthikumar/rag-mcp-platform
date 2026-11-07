import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.admin import audit_index, maintain_document, reconcile_sources


class IngestionAdminTests(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock()
        self.conn = self.pool.connection.return_value.__enter__.return_value
        self.cur = self.conn.cursor.return_value
        patcher = patch("rag_mcp_platform.ingest.admin.get_pool", return_value=self.pool)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = SimpleNamespace(embedding_signature="model:2", processing_fingerprint="pipeline:7", embedding_dim=2)

    def test_audit_reports_issues_without_sensitive_source_labels(self):
        self.cur.fetchall.return_value = [{
            "id": "document", "source_path": "temp/rag-mcp-platform-ingest-old/alice@example.org.txt",
            "embedding_signature": None, "metadata": {}, "chunk_count": 0,
            "missing_vectors": 0, "dimensions": [3],
        }]
        with patch("rag_mcp_platform.ingest.admin.get_settings", return_value=self.settings):
            report = audit_index()
        self.assertEqual(report["documents_with_issues"], 1)
        self.assertIn("legacy_temporary_upload_identity", report["findings"][0]["issues"])
        self.assertNotIn("alice@example.org", json.dumps(report))
        self.conn.execute.assert_called_once_with("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

    def test_pii_audit_does_not_echo_stored_values(self):
        self.cur.fetchall.return_value = [{
            "id": "document", "source_path": "guide.txt", "embedding_signature": "model:2",
            "metadata": {"processing_fingerprint": "pipeline:7"}, "chunk_count": 1,
            "missing_vectors": 0, "dimensions": [2],
        }]
        self.cur.fetchmany.side_effect = [[{
            "id": "document", "source_path": "guide.txt", "title": "Guide", "url": None,
            "doc_type": "text", "document_metadata": {}, "chunk_metadata": {}, "content": "alice@example.org",
        }], []]
        with patch("rag_mcp_platform.ingest.admin.get_settings", return_value=self.settings):
            report = audit_index(check_pii=True)
        self.assertIn("pii_policy_rejection", report["findings"][0]["issues"])
        self.assertNotIn("alice@example.org", json.dumps(report))

    def test_delete_is_read_only_without_apply(self):
        self.cur.fetchone.return_value = {"id": "document", "source_path": "guide.txt", "version": 2}
        report = maintain_document("document", 2)
        self.assertFalse(report["applied"])
        self.conn.execute.assert_called_once_with("SET TRANSACTION READ ONLY")

    def test_apply_requires_matching_revision(self):
        self.cur.fetchone.return_value = {"id": "document", "source_path": "guide.txt", "version": 3}
        with self.assertRaisesRegex(ValueError, "revision changed"):
            maintain_document("document", 2, apply=True)
        self.assertFalse(any(call.args[0].startswith("DELETE") for call in self.conn.execute.call_args_list))

    def test_rename_apply_updates_identity_and_chunk_metadata(self):
        self.cur.fetchone.side_effect = [{"id": "document", "source_path": "old.txt", "version": 2}, None]
        report = maintain_document("document", 2, rename_to="new.txt", apply=True)
        self.assertTrue(report["applied"])
        self.assertEqual(report["action"], "rename")
        queries = [call.args[0] for call in self.conn.execute.call_args_list]
        self.assertTrue(any(query.startswith("UPDATE documents") for query in queries))
        self.assertTrue(any(query.startswith("UPDATE chunks") for query in queries))

    def test_reconciliation_previews_only_missing_sources_under_root(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "present.txt"
            existing.write_text("Text.", encoding="utf-8")
            self.cur.fetchall.return_value = [
                {"id": "missing", "source_path": str(Path(directory) / "missing.txt"), "version": 1},
                {"id": "present", "source_path": str(existing), "version": 1},
                {"id": "upload", "source_path": "upload://guide.txt", "version": 1},
            ]
            report = reconcile_sources(directory)
        self.assertEqual([item["document_id"] for item in report["missing_sources"]], ["missing"])
        self.assertFalse(report["applied"])


if __name__ == "__main__":
    unittest.main()