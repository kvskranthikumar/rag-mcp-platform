import json
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.ingest.indexer import (
    _finish_run, _record_file_result, _start_run, ingest, ingest_files, ingest_stream,
)
from rag_mcp_platform.schema import _ddl_statements


class IngestionOutcomeTests(unittest.TestCase):
    def setUp(self):
        settings = SimpleNamespace(
            embedding_dim=2, embedding_signature="test:2", docs_source_dir="docs"
        )
        patches = {
            "get_settings": {"return_value": settings},
            "get_embedding_provider": {"return_value": SimpleNamespace(dimension=2)},
            "get_pool": {},
            "_start_run": {"return_value": 7},
            "_record_file_result": {},
            "_finish_run": {},
            "_process_file": {},
        }
        self.mocks = {}
        for name, options in patches.items():
            patcher = patch(f"rag_mcp_platform.ingest.indexer.{name}", **options)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)
        self.process = self.mocks["_process_file"]
        self.record = self.mocks["_record_file_result"]
        self.finish = self.mocks["_finish_run"]

    def assert_finished(self, status, seen, changed, written, failed):
        self.finish.assert_called_once()
        self.assertEqual(self.finish.call_args.args[1:6], (7, seen, changed, written, status))
        self.assertEqual(self.finish.call_args.kwargs, {"failed": failed})

    def test_mixed_results_are_partial_success(self):
        self.process.side_effect = [("indexed", 3), ValueError("bad embedding"), ("skipped", 0)]
        summary = ingest_files(["one.md", "two.md", "three.md"])
        self.assertEqual(summary["status"], "partial_success")
        self.assertEqual(summary["files_failed"], 1)
        self.assertEqual(summary["results"][1]["error"], "bad embedding")
        self.assert_finished("partial_success", 3, 1, 3, 1)
        self.assertEqual(self.record.call_count, 3)

    def test_all_failures_are_failed(self):
        self.process.side_effect = [ValueError("one"), RuntimeError("two")]
        self.assertEqual(ingest_files(["one.md", "two.md"])["status"], "failed")
        self.assert_finished("failed", 2, 0, 0, 2)

    def test_rejected_files_are_not_successful(self):
        self.process.side_effect = [("empty", 0), ("unsupported", 0)]
        summary = ingest_files(["empty.md", "unsupported.bin"])
        self.assertEqual(summary["files_failed"], 2)
        self.assertTrue(all(result.get("error") for result in summary["results"]))
        self.assert_finished("failed", 2, 0, 0, 2)

    def test_skipped_and_duplicate_files_are_successful(self):
        self.process.side_effect = [("skipped", 0), ("duplicate", 0)]
        self.assertEqual(ingest_files(["one.md", "two.md"])["status"], "success")
        self.assert_finished("success", 2, 0, 0, 0)

    def test_empty_batch_is_successful_noop(self):
        summary = ingest_files([])
        self.assertEqual(summary["results"], [])
        self.assert_finished("success", 0, 0, 0, 0)

    def test_result_is_durable_before_done_event(self):
        self.process.return_value = ("indexed", 2)
        stream = ingest_stream(["temp/guide.md"], source_paths={"temp/guide.md": "upload://guide.md"})
        self.addCleanup(stream.close)
        self.assertEqual(next(stream)["phase"], "start")
        self.assertEqual(next(stream)["phase"], "done")
        self.assertEqual(self.record.call_args.args[2]["source_path"], "upload://guide.md")
        self.assertEqual(self.record.call_count, 1)
        stream.close()
        self.assert_finished("success", 1, 1, 2, 0)

    def test_close_at_start_marks_file_and_run_interrupted(self):
        stream = ingest_stream(["one.md", "two.md"])
        next(stream)
        stream.close()
        self.process.assert_not_called()
        self.assertEqual(self.record.call_args.args[2]["status"], "interrupted")
        self.assert_finished("interrupted", 1, 0, 0, 1)

    def test_close_between_files_preserves_completed_results(self):
        self.process.return_value = ("indexed", 2)
        stream = ingest_stream(["one.md", "two.md"])
        next(stream)
        next(stream)
        stream.close()
        self.assert_finished("interrupted", 1, 1, 2, 0)
        self.assertEqual(self.record.call_count, 1)

    def test_keyboard_interrupt_is_recorded_and_propagated(self):
        self.process.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            ingest_files(["one.md"])
        self.assert_finished("interrupted", 1, 0, 0, 1)
        self.assertEqual(self.record.call_args.args[2]["error"], "KeyboardInterrupt")

    def test_cli_retains_fail_fast_behavior_and_records_failure(self):
        self.process.side_effect = [("indexed", 2), ValueError("invalid"), ("indexed", 5)]
        with patch("rag_mcp_platform.ingest.indexer.iter_files", return_value=iter([
            Path("one.md"), Path("two.md"), Path("three.md")
        ])):
            with self.assertRaisesRegex(ValueError, "invalid"):
                ingest()
        self.assertEqual(self.process.call_count, 2)
        self.assertEqual(self.record.call_count, 2)
        self.assert_finished("failed", 2, 1, 2, 1)

    def test_cli_returns_aggregate_outcome(self):
        self.process.return_value = ("indexed", 2)
        with patch("rag_mcp_platform.ingest.indexer.iter_files", return_value=iter([Path("one.md")])):
            summary = ingest()
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["chunks_written"], 2)
        self.assert_finished("success", 1, 1, 2, 0)

    def test_file_discovery_failure_finalizes_run(self):
        def paths():
            raise OSError("cannot list directory")
            yield

        with patch("rag_mcp_platform.ingest.indexer.iter_files", return_value=paths()):
            with self.assertRaisesRegex(OSError, "cannot list directory"):
                ingest()
        self.assert_finished("failed", 0, 0, 0, 0)

    def test_run_start_failure_prevents_document_processing(self):
        self.mocks["_start_run"].side_effect = RuntimeError("schema upgrade required")
        with self.assertRaisesRegex(RuntimeError, "schema upgrade required"):
            ingest_files(["one.md"])
        self.process.assert_not_called()
        self.finish.assert_not_called()


class IngestionRunPersistenceTests(unittest.TestCase):
    def test_run_creation_requires_upgraded_columns(self):
        pool = MagicMock()
        _start_run(pool)
        cursor = pool.connection.return_value.__enter__.return_value.cursor.return_value
        self.assertIn("(files_failed, results, trace_id)", cursor.execute.call_args.args[0])

    def test_run_creation_persists_trace_id(self):
        pool = MagicMock()
        _start_run(pool, "a" * 32)
        cursor = pool.connection.return_value.__enter__.return_value.cursor.return_value
        self.assertEqual(cursor.execute.call_args.args[1], ("a" * 32,))

    def test_file_result_updates_durable_progress(self):
        pool = MagicMock()
        result = {"file": "one.md", "status": "failed", "chunks": 0, "error": "invalid"}
        _record_file_result(pool, 7, result)
        query = pool.connection.return_value.__enter__.return_value.execute.call_args
        self.assertIn("results = results || %s::jsonb", query.args[0])
        self.assertEqual(query.args[1][:3], (0, 0, 1))
        self.assertEqual(json.loads(query.args[1][3]), [result])

    def test_finalization_persists_failed_count(self):
        pool = MagicMock()
        _finish_run(pool, 7, 2, 1, 3, "partial_success", "one failure", failed=1)
        query = pool.connection.return_value.__enter__.return_value.execute.call_args
        self.assertIn("finished_at = now()", query.args[0])
        self.assertEqual(query.args[1], (2, 1, 3, "partial_success", "one failure", 1, 7))

    def test_schema_upgrade_is_additive(self):
        statements = _ddl_statements(2)
        self.assertTrue(any("ADD COLUMN IF NOT EXISTS files_failed" in stmt for stmt in statements))
        self.assertTrue(any("ADD COLUMN IF NOT EXISTS results" in stmt for stmt in statements))
        self.assertTrue(any("ADD COLUMN IF NOT EXISTS trace_id" in stmt for stmt in statements))


class IngestionCliOutcomeTests(unittest.TestCase):
    def test_unsuccessful_batch_exits_nonzero_after_printing_summary(self):
        from rag_mcp_platform.ingest.cli import main

        for outcome in ("partial_success", "failed"):
            with self.subTest(outcome=outcome):
                output = StringIO()
                with (
                    patch.object(sys, "argv", ["rag-mcp-platform", "ingest"]),
                    patch("rag_mcp_platform.ingest.indexer.ingest", return_value={"status": outcome}),
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main()
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(json.loads(output.getvalue())["status"], outcome)

    def test_successful_batch_does_not_raise(self):
        from rag_mcp_platform.ingest.cli import main

        with (
            patch.object(sys, "argv", ["rag-mcp-platform", "ingest"]),
            patch("rag_mcp_platform.ingest.indexer.ingest", return_value={"status": "success"}),
            redirect_stdout(StringIO()),
        ):
            main()


class IngestionUiOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.streamlit = MagicMock()
        self.streamlit.session_state = {"_status": {
            "documents": 0, "chunks": 0, "mixed": False, "stale": False,
        }}
        self.streamlit.file_uploader.return_value = None
        self.streamlit.button.return_value = False
        self.streamlit.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]
        with patch.dict(sys.modules, {"streamlit": self.streamlit}):
            self.app = runpy.run_module("rag_mcp_platform.ingest_ui.app")
        self.streamlit.reset_mock()

    def test_rejected_batch_is_not_rendered_as_success(self):
        self.app["_render_results"]([{"file": "one.md", "status": "empty", "chunks": 0}])
        self.streamlit.error.assert_called_once()
        self.streamlit.success.assert_not_called()

    def test_mixed_batch_renders_warning(self):
        self.app["_render_results"]([
            {"file": "one.md", "status": "indexed", "chunks": 2},
            {"file": "two.md", "status": "failed", "chunks": 0},
        ])
        self.streamlit.warning.assert_called_once()
        self.streamlit.success.assert_not_called()

    def test_run_error_prevents_success_even_with_completed_files(self):
        self.app["_render_results"](
            [{"file": "one.md", "status": "indexed", "chunks": 2}], "logging unavailable"
        )
        self.streamlit.error.assert_called_once_with("Ingestion stopped: logging unavailable")
        self.streamlit.success.assert_not_called()

    def test_interrupted_ui_retains_error_and_marks_unfinished_files(self):
        def events():
            yield {"phase": "start", "index": 1, "file": "one.md"}
            yield {"phase": "done", "index": 1, "file": "one.md", "status": "indexed", "chunks": 2}
            yield {"phase": "start", "index": 2, "file": "two.md"}
            raise RuntimeError("connection lost")

        self.streamlit.session_state["uploader_key"] = 0
        with patch.dict(self.app, {
            "check_connection": MagicMock(return_value=(True, "OK")),
            "stage_uploads": MagicMock(return_value={
                "temp/one.md": "upload://one.md", "temp/two.md": "upload://two.md",
            }),
            "ingest_stream": MagicMock(return_value=events()),
        }):
            self.app["_run_ingest"]([MagicMock(), MagicMock()], False)
        results = self.streamlit.session_state["last_results"]
        self.assertEqual(results[0]["status"], "indexed")
        self.assertEqual(results[1]["status"], "interrupted")
        self.assertEqual(self.streamlit.session_state["last_run_error"], "connection lost")
        self.assertFalse(self.streamlit.session_state["ingesting"])

    def test_original_upload_staging_is_deleted_after_success_or_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                staged = []
                uploaded = BytesIO(b"Technical documentation.")
                uploaded.name = "guide.txt"

                def events(paths, **kwargs):
                    staged.extend(Path(path) for path in paths)
                    self.assertTrue(all(path.exists() for path in staged))
                    yield {"phase": "start", "index": 1, "file": "guide.txt"}
                    if fail:
                        raise RuntimeError("processing failed")
                    yield {"phase": "done", "index": 1, "file": "guide.txt", "status": "indexed", "chunks": 1}

                self.streamlit.session_state["uploader_key"] = 0
                with patch.dict(self.app, {
                    "check_connection": MagicMock(return_value=(True, "OK")),
                    "ingest_stream": events,
                }):
                    self.app["_run_ingest"]([uploaded], False)
                self.assertTrue(staged)
                self.assertTrue(all(not path.exists() for path in staged))
                self.assertNotIn("Technical documentation.", json.dumps(self.streamlit.session_state))


if __name__ == "__main__":
    unittest.main()