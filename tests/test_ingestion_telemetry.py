import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from rag_mcp_platform.core.search import answer, search
from rag_mcp_platform.ingest.indexer import ingest_files
from rag_mcp_platform.telemetry import (
    capture_metrics, increment_metric, measured, observation, record_usage,
)


class FakeLangfuseClient:
    def __init__(self):
        self.observations = []

    @contextmanager
    def start_as_current_observation(self, **options):
        item = {"options": options, "updates": []}
        self.observations.append(item)
        span = SimpleNamespace(update=lambda **kwargs: item["updates"].append(kwargs))
        yield span


class IngestionTelemetryTests(unittest.TestCase):
    def test_metrics_capture_success_and_failure_without_source_content(self):
        with capture_metrics() as metrics:
            measured("extraction", lambda: None)
            with self.assertRaises(ValueError):
                measured("pii_screening", lambda: (_ for _ in ()).throw(ValueError("private source text")))
            increment_metric("embedding_sdk_requests")
        self.assertIn("total_seconds", metrics)
        self.assertIn("extraction", metrics["stage_seconds"])
        self.assertIn("pii_screening", metrics["stage_seconds"])
        self.assertNotIn("private source text", json.dumps(metrics))

    def test_nested_metrics_are_isolated(self):
        with capture_metrics() as outer:
            increment_metric("requests", 1)
            with capture_metrics() as inner:
                increment_metric("requests", 3)
            increment_metric("requests", 1)
        self.assertEqual(outer["requests"], 2)
        self.assertEqual(inner["requests"], 3)

    def test_provider_usage_is_recorded_from_response(self):
        from test_embedding_batches import EmbeddingBatchTests

        provider, client, _ = EmbeddingBatchTests().provider()
        client.embeddings.create.side_effect = None
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
            usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
        )
        with capture_metrics() as metrics:
            provider.embed_documents(["text"])
        self.assertEqual(metrics["embedding_sdk_requests"], 1)
        self.assertEqual(metrics["embedding_prompt_tokens"], 7)
        self.assertEqual(metrics["embedding_total_tokens"], 7)

    def test_observations_record_only_supplied_safe_metadata(self):
        client = FakeLangfuseClient()
        with patch("rag_mcp_platform.telemetry.get_langfuse_client", return_value=client):
            with observation(
                "retrieval.search", trace_id="a" * 32,
                metadata={"query_characters": 42},
            ) as span:
                span.update(output={"result_count": 3})
        recorded = client.observations[0]
        self.assertEqual(recorded["options"]["trace_context"], {"trace_id": "a" * 32})
        self.assertEqual(recorded["options"]["metadata"], {"query_characters": 42})
        self.assertEqual(recorded["updates"], [{"output": {"result_count": 3}}])
        self.assertNotIn("input", recorded["options"])

    def test_observation_errors_exclude_exception_messages_and_preserve_failure(self):
        client = FakeLangfuseClient()
        with patch("rag_mcp_platform.telemetry.get_langfuse_client", return_value=client):
            with self.assertRaisesRegex(ValueError, "private source text"):
                with observation("ingestion.extraction"):
                    raise ValueError("private source text")
        self.assertEqual(
            client.observations[0]["updates"],
            [{"level": "ERROR", "status_message": "ValueError"}],
        )
        self.assertNotIn("private source text", json.dumps(client.observations))

    def test_token_usage_attaches_to_current_observation(self):
        client = FakeLangfuseClient()
        with patch("rag_mcp_platform.telemetry.get_langfuse_client", return_value=client):
            with observation("answer.generation"):
                record_usage(input_tokens=12, output_tokens=5, total_tokens=17)
        self.assertEqual(
            client.observations[0]["updates"],
            [{"usage_details": {"input": 12, "output": 5, "total": 17}}],
        )

    def test_search_trace_omits_query_content(self):
        client = FakeLangfuseClient()
        with (
            patch("rag_mcp_platform.telemetry.get_langfuse_client", return_value=client),
            patch("rag_mcp_platform.core.search.get_settings", return_value=SimpleNamespace(top_k=8)),
            patch("rag_mcp_platform.core.search._search", return_value=[object()]),
        ):
            results = search("private query", 4)
        self.assertEqual(len(results), 1)
        self.assertEqual(client.observations[0]["options"]["metadata"], {
            "query_characters": len("private query"), "top_k": 4,
        })
        self.assertNotIn("private query", json.dumps(client.observations))

    def test_answer_traces_omit_query_context_and_answer(self):
        client = FakeLangfuseClient()
        chunk = SimpleNamespace(
            citation=lambda: "Guide > Secret",
            content="private document passage",
            source_path="private/path.md",
            url=None,
            score=0.9,
        )

        def complete(*args, **kwargs):
            record_usage(input_tokens=9, output_tokens=4, total_tokens=13)
            return "private model answer"

        settings = SimpleNamespace(
            top_k=8, llm_temperature=0.1, llm_model="test-model", llm_provider="openai",
        )
        with (
            patch("rag_mcp_platform.telemetry.get_langfuse_client", return_value=client),
            patch("rag_mcp_platform.core.search.get_settings", return_value=settings),
            patch("rag_mcp_platform.core.search.search", return_value=[chunk]),
            patch("rag_mcp_platform.providers.llm.get_llm_provider", return_value=SimpleNamespace(complete=complete)),
        ):
            result = answer("private query", 2)
        self.assertEqual(result["answer"], "private model answer")
        serialized = json.dumps(client.observations)
        for secret in ("private query", "private document passage", "private model answer", "private/path.md"):
            self.assertNotIn(secret, serialized)
        generation = next(item for item in client.observations if item["options"]["name"] == "answer.generation")
        self.assertEqual(generation["updates"], [
            {"usage_details": {"input": 9, "output": 4, "total": 13}},
        ])

    def test_failed_file_metrics_are_in_persisted_result(self):
        settings = SimpleNamespace(embedding_dim=2, embedding_signature="test:2")

        def process(*args, **kwargs):
            measured("extraction", lambda: None)
            raise ValueError("bad extraction")

        with (
            patch("rag_mcp_platform.ingest.indexer.get_settings", return_value=settings),
            patch("rag_mcp_platform.ingest.indexer.get_embedding_provider", return_value=SimpleNamespace(dimension=2)),
            patch("rag_mcp_platform.ingest.indexer.get_pool"),
            patch("rag_mcp_platform.ingest.indexer._start_run"),
            patch("rag_mcp_platform.ingest.indexer._finish_run"),
            patch("rag_mcp_platform.ingest.indexer._record_file_result") as record,
            patch("rag_mcp_platform.ingest.indexer._process_file", side_effect=process),
        ):
            report = ingest_files(["guide.md"])
        self.assertEqual(report["status"], "failed")
        self.assertIn("extraction", record.call_args.args[2]["metrics"]["stage_seconds"])
        self.assertIn("total_seconds", record.call_args.args[2]["metrics"])


if __name__ == "__main__":
    unittest.main()