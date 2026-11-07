import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_mcp_platform.providers.embeddings.openai_provider import OpenAIEmbeddingProvider


class EmbeddingBatchTests(unittest.TestCase):
    def provider(self, **options):
        client = MagicMock()
        with patch("openai.OpenAI", return_value=client) as constructor:
            provider = OpenAIEmbeddingProvider(
                "text-embedding-3-small", 2, api_key="test-only-key", **options
            )
        provider._encoding = SimpleNamespace(encode=lambda text, **kwargs: list(text))

        def create(**kwargs):
            return SimpleNamespace(data=[
                SimpleNamespace(index=index, embedding=[float(index), float(len(text))])
                for index, text in reversed(list(enumerate(kwargs["input"])))
            ])

        client.embeddings.create.side_effect = create
        return provider, client, constructor

    def test_batch_is_bounded_by_tokens_and_item_count(self):
        provider, client, _ = self.provider(batch_size=2, batch_max_tokens=6)
        vectors = provider.embed_documents(["aaa", "bbb", "cccc", "dd", "eee"])
        self.assertEqual(len(vectors), 5)
        batches = [call.kwargs["input"] for call in client.embeddings.create.call_args_list]
        self.assertEqual(batches, [["aaa", "bbb"], ["cccc", "dd"], ["eee"]])
        self.assertTrue(all(sum(map(len, batch)) <= 6 and len(batch) <= 2 for batch in batches))

    def test_response_order_is_restored_using_indices(self):
        provider, _, _ = self.provider()
        self.assertEqual(provider.embed_documents(["a", "bb"]), [[0.0, 1.0], [1.0, 2.0]])

    def test_duplicate_inputs_are_embedded_once_per_call(self):
        provider, client, _ = self.provider()
        vectors = provider.embed_documents(["same", "other", "same"])
        self.assertEqual(client.embeddings.create.call_args.kwargs["input"], ["same", "other"])
        self.assertEqual(vectors[0], vectors[2])
        self.assertEqual(len(vectors), 3)

    def test_all_inputs_are_validated_before_any_request(self):
        for invalid in ("", "  ", None, "too long"):
            with self.subTest(invalid=invalid):
                provider, client, _ = self.provider(max_input_tokens=5)
                with self.assertRaises(ValueError):
                    provider.embed_documents(["good", invalid])
                client.embeddings.create.assert_not_called()

    def test_batch_budget_smaller_than_one_input_is_rejected(self):
        provider, client, _ = self.provider(batch_max_tokens=3)
        with self.assertRaisesRegex(ValueError, "token budget"):
            provider.embed_documents(["four"])
        client.embeddings.create.assert_not_called()

    def test_missing_duplicate_or_invalid_response_indices_fail(self):
        for indices in ([], [0], [0, 0], [0, 2], [True, 1]):
            with self.subTest(indices=indices):
                provider, client, _ = self.provider()
                client.embeddings.create.side_effect = None
                client.embeddings.create.return_value = SimpleNamespace(data=[
                    SimpleNamespace(index=index, embedding=[0.1, 0.2]) for index in indices
                ])
                with self.assertRaisesRegex(ValueError, "response indices"):
                    provider.embed_documents(["one", "two"])

    def test_sdk_receives_retry_and_timeout_limits(self):
        _, _, constructor = self.provider(max_retries=3, timeout=20.0)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 3)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 20.0)

    def test_azure_sdk_receives_the_same_limits(self):
        with patch("openai.AzureOpenAI") as constructor:
            OpenAIEmbeddingProvider(
                "text-embedding-3-small", 2, azure=True,
                azure_endpoint="https://example.openai.azure.com", azure_deployment="test",
                max_retries=3, timeout=20.0,
            )
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 3)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 20.0)

    def test_invalid_configuration_fails_before_client_creation(self):
        for options in ({"batch_size": 0}, {"max_retries": -1}, {"timeout": float("nan")}, {"max_input_tokens": 0}):
            with self.subTest(options=options), patch("openai.OpenAI") as constructor:
                with self.assertRaises(ValueError):
                    OpenAIEmbeddingProvider("text-embedding-3-small", 2, **options)
                constructor.assert_not_called()

    def test_empty_batch_makes_no_request(self):
        provider, client, _ = self.provider()
        self.assertEqual(provider.embed_documents([]), [])
        client.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()