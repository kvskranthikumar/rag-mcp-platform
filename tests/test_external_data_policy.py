import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_mcp_platform.providers.embeddings import get_embedding_provider
from rag_mcp_platform.providers.llm import get_llm_provider
from rag_mcp_platform.mcp_server.server import _require_http_auth
from rag_mcp_platform.settings import Settings


class ExternalDataPolicyTests(unittest.TestCase):
    def setUp(self):
        get_embedding_provider.cache_clear()
        get_llm_provider.cache_clear()

    def tearDown(self):
        get_embedding_provider.cache_clear()
        get_llm_provider.cache_clear()

    def test_hosted_embeddings_are_blocked_without_opt_in(self):
        settings = Settings(embedding_provider="azure", allow_external_data=False)
        with patch("rag_mcp_platform.providers.embeddings.get_settings", return_value=settings):
            with self.assertRaisesRegex(PermissionError, "ALLOW_EXTERNAL_DATA=true"):
                get_embedding_provider()

    def test_hosted_answer_provider_is_blocked_without_opt_in(self):
        settings = Settings(llm_provider="gemini", allow_external_data=False)
        with patch("rag_mcp_platform.providers.llm.get_settings", return_value=settings):
            with self.assertRaisesRegex(PermissionError, "ALLOW_EXTERNAL_DATA=true"):
                get_llm_provider()

    def test_non_loopback_ollama_is_blocked_without_opt_in(self):
        settings = Settings(
            llm_provider="ollama",
            ollama_base_url="http://inference.internal:11434",
            allow_external_data=False,
        )
        with patch("rag_mcp_platform.providers.llm.get_settings", return_value=settings):
            with self.assertRaisesRegex(PermissionError, "ALLOW_EXTERNAL_DATA=true"):
                get_llm_provider()

    def test_loopback_ollama_is_allowed_without_opt_in(self):
        settings = Settings(llm_provider="ollama", allow_external_data=False)
        with patch("rag_mcp_platform.providers.llm.get_settings", return_value=settings):
            self.assertEqual(get_llm_provider().name, f"ollama:{settings.llm_model}")

    def test_exposed_mcp_http_requires_api_key(self):
        with self.assertRaisesRegex(PermissionError, "MCP_API_KEY"):
            _require_http_auth(Settings(mcp_host="0.0.0.0", mcp_api_key=None))

    def test_loopback_mcp_http_does_not_require_api_key(self):
        _require_http_auth(Settings(mcp_host="127.0.0.1", mcp_api_key=None))

    def test_local_embedding_does_not_execute_remote_model_code(self):
        from unittest.mock import Mock

        from rag_mcp_platform.providers.embeddings.local import LocalEmbeddingProvider

        model = Mock()
        model.get_sentence_embedding_dimension.return_value = 2
        constructor = Mock(return_value=model)
        module = SimpleNamespace(SentenceTransformer=constructor)
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            LocalEmbeddingProvider(model="sample-model", dimension=2)
        constructor.assert_called_once_with("sample-model", trust_remote_code=False)


if __name__ == "__main__":
    unittest.main()