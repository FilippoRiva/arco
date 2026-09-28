import sys
from types import ModuleType, SimpleNamespace

from arco.core import llm_tools


def test_batch_check_groups_models_by_provider(monkeypatch):
    catalog_by_provider = {
        "openai": frozenset({"model-a", "model-b"}),
        "ollama": frozenset({"llama3.2"}),
    }
    requested_providers = []

    def get_catalog(provider: str) -> frozenset[str]:
        requested_providers.append(provider)
        return catalog_by_provider[provider]

    monkeypatch.setattr(llm_tools, "_get_provider_model_catalog", get_catalog)

    available, message = llm_tools.check_models_availability(
        [
            ("openai", "model-a"),
            ("openai", "model-b"),
            ("ollama", "llama3.2:latest"),
        ]
    )

    assert available
    assert "openai" in message and "ollama" in message
    assert requested_providers == ["ollama", "openai"]


def test_model_catalog_is_fetched_once_per_provider_and_ttl(monkeypatch):
    model_list_calls = 0

    class FakeOpenAIError(Exception):
        pass

    class FakeModels:
        def list(self):
            nonlocal model_list_calls
            model_list_calls += 1
            return [SimpleNamespace(id="model-a"), SimpleNamespace(id="model-b")]

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = FakeOpenAI
    openai_stub.OpenAIError = FakeOpenAIError

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", openai_stub)
    monkeypatch.setattr(llm_tools, "_model_catalog_ttl_bucket", lambda: 1)
    llm_tools._fetch_provider_model_catalog.cache_clear()

    try:
        available, _ = llm_tools.check_models_availability(
            [("openai", "model-a"), ("openai", "model-b")]
        )
        assert available
        assert model_list_calls == 1

        available, _ = llm_tools.check_model_availability("openai", "model-a")
        assert available
        assert model_list_calls == 1

        available, message = llm_tools.check_model_availability("openai", "missing")
        assert not available
        assert "missing" in message
        assert model_list_calls == 1

        monkeypatch.setattr(llm_tools, "_model_catalog_ttl_bucket", lambda: 2)
        available, _ = llm_tools.check_model_availability("openai", "model-a")
        assert available
        assert model_list_calls == 2
    finally:
        llm_tools._fetch_provider_model_catalog.cache_clear()
