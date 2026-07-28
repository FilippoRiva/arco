import json
import os
import re
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .tracking import LLMCallAccumulator

# Global parameters
OLLAMA_REQUEST_TIMEOUT: int = 600
OLLAMA_URL: str = "http://localhost:11434"

DEFAULT_LLM_ACC = LLMCallAccumulator("None")

if TYPE_CHECKING:
    from .agent_config import AgentConfig


import logging

logger = logging.getLogger(__name__)


class LLMAnswer:
    """Pre-extracted LLM response — no manual content/logprobs wrangling."""

    def __init__(self, response):
        self.text: str = (
            str(response.content) if hasattr(response, "content") else str(response)
        )
        self.logprobs: list[tuple[str, float | int]] = _extract_logprobs(response)

    def extract_fenced_content(self):
        """
        Extracts content from a fenced block.
        If no fence exists, uses the raw text.
        """
        fence_re = re.compile(r"```[^\n]*\n?(.*?)\n?```", re.DOTALL)

        match = fence_re.search(self.text)
        content = match.group(1).strip() if match else self.text.strip()
        return content

    def extract_json(self):
        content = self.extract_fenced_content()
        try:
            return json.loads(content)
        except JSONDecodeError:
            return {}

    def extract_json_list(self):
        content = self.extract_fenced_content()
        try:
            json_list = json.loads(content)
            if not isinstance(json_list, list):
                raise TypeError("Parsed json didn't produce a list")
            return json_list
        except JSONDecodeError, TypeError:
            return []

    def extract_python(self):
        return self.extract_fenced_content()

    def extract_sql(self):
        return self.extract_fenced_content()


class LLM:
    _REFINEMENT_SUFFIX = """
    ## ITERATIVE REFINEMENT
    Your previous attempt produced the following response:
    ---
    {previous_response}
    ---
    Carefully review your previous response.
    - If it is correct and complete, reproduce it exactly (same content, same format).
    - If you identify errors or improvements, output a revised version.
    Output only the final response with no meta-commentary.
    """

    _ERROR_SUFFIX = """
    ## ITERATIVE REFINEMENT — EXECUTION ERROR
    Your previous attempt produced the following response:
    ---
    {previous_response}
    ---
    When executed, it raised the following error:
    ---
    {execution_error}
    ---
    You MUST fix this error. Output only the corrected response with no meta-commentary.
    """

    def __init__(self, base_chat_model: BaseChatModel):
        self._chat_model: BaseChatModel = base_chat_model
        self.cot_enabled: bool = False
        self.last_answer: LLMAnswer | None = None
        self.execution_error: str | None = None

    def invoke(self, prompt: str) -> LLMAnswer:
        if self.cot_enabled:
            answer = self._cot_invoke(prompt, self.execution_error)
        else:
            answer = LLMAnswer(self._chat_model.invoke(prompt))
        self.last_answer = answer
        return answer

    def _cot_invoke(self, prompt: str, execution_error: str | None) -> LLMAnswer:
        if execution_error:
            suffix = self._ERROR_SUFFIX.format(
                previous_response=self.last_answer.text,
                execution_error=execution_error,
            )
        else:
            suffix = self._REFINEMENT_SUFFIX.format(
                previous_response=self._previous_response,
            )
        return LLMAnswer(self._chat_model.invoke(prompt + suffix))


def get_llm_from_config(agent_config: AgentConfig, llm_acc: LLMCallAccumulator) -> LLM:
    temp, top_p, top_k = agent_config.get_candidate_params()[0]

    return get_llm(
        provider=agent_config.provider,
        model=agent_config.model,
        max_tokens=agent_config.max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        num_beams=agent_config.num_beams,
        no_repeat_ngram_size=agent_config.no_repeat_ngram_size,
        llm_accumulator=llm_acc,
    )


def get_llm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    streaming=True,
    max_tokens: int = 2000,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    num_beams: int | None = None,
    no_repeat_ngram_size: int | None = None,
    llm_accumulator: LLMCallAccumulator = DEFAULT_LLM_ACC,
    openrouter_url: str = "https://openrouter.ai/api/v1",
) -> LLM:
    """Factory method to create LLM instances with specific parameters.

    Creates a new LLM instance instead of mutating the global self.llm,
    which allows per-step parameter customization.

    Args:
        temperature: Sampling temperature
        max_tokens: Maximum tokens for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter (skipped for OpenAI)
        num_beams: Beam search width, 1 = greedy/disabled (skipped for OpenAI)
        no_repeat_ngram_size: Prevent repeating n-grams of this size (skipped for OpenAI)
        streaming: Whether to stream the response tokens in real-time.
        provider: The LLM provider to use (e.g., 'openai', 'ollama', 'anthropic').
        model: The specific model ID/name to instantiate.
        llm_accumulator: An instance to track or log LLM calls and usage.
        openrouter_url: Base URL for the Openrouter API, if using openrouter provider.

    Returns:
        BaseChatModel: A configured instance of a LangChain-compatible chat model.
    """
    if provider.lower() == "openai":
        chat_model = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            callbacks=[llm_accumulator],
            top_p=top_p,
            logprobs=True,
        )
    elif provider.lower() == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter requires an API key: pass openrouter_api_key or "
                "set the OPENROUTER_API_KEY environment variable."
            )
        chat_model = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=openrouter_url,
            temperature=temperature,
            # max_tokens=max_tokens,
            # streaming=streaming,
            callbacks=[llm_accumulator],
            # top_p=top_p,
            logprobs=True,
            extra_body={
                "provider": {
                    "require_parameters": True  # use only providers that allow all the parameters from the request
                }
            },
        )
    else:
        kwargs = {
            "model": model,
            "base_url": OLLAMA_URL,
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "client_kwargs": {"timeout": OLLAMA_REQUEST_TIMEOUT},
            "callbacks": [llm_accumulator],
            "logprobs": True,
        }

        if top_k is not None:
            kwargs["top_k"] = top_k
        if num_beams is not None and num_beams > 1:
            kwargs["num_beams"] = num_beams
        if no_repeat_ngram_size is not None:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        chat_model = ChatOllama(**kwargs)
    return LLM(base_chat_model=chat_model)


def _extract_logprobs(message: AIMessage) -> list[tuple[str, float | int]] | None:
    metadata = message.response_metadata
    if "logprobs" in metadata and metadata["logprobs"] is not None:
        logprobs_data = metadata["logprobs"]

        # OPENAI / OPENROUTER
        if isinstance(logprobs_data, dict) and "content" in logprobs_data:
            content_logprobs = logprobs_data.get("content") or []
            token_logprob_tuple_list = [
                (token_info.get("token"), token_info.get("logprob"))
                for token_info in content_logprobs
                if "logprob" in token_info
            ]

            if "deepseek" in metadata["model_name"]:
                think_end = "</think>"
                end_token = "<｜end▁of▁sentence｜>"  # Cleaned spacing
                tokens = [item[0] for item in token_logprob_tuple_list]
                start_idx = 0
                if think_end in tokens:
                    start_idx = tokens.index(think_end) + 1
                end_idx = len(token_logprob_tuple_list)
                if end_token in tokens:
                    end_idx = tokens.index(end_token)
                token_logprob_tuple_list = token_logprob_tuple_list[start_idx:end_idx]

            return token_logprob_tuple_list
        # OLLAMA
        elif isinstance(logprobs_data, list) and len(logprobs_data) > 0:
            if "gemma4" in metadata["model"]:
                # manually excluding thinking tokens
                end_token = "<channel|>"
                tokens = [logprobs_data[i]["token"] for i in range(len(logprobs_data))]
                end_of_thinking_token_index = tokens.index(end_token)
                return [
                    (logprobs_data[i]["token"], logprobs_data[i]["logprob"])
                    for i in range(end_of_thinking_token_index + 1, len(logprobs_data))
                ]

            return [
                (logprobs_data[i]["token"], logprobs_data[i]["logprob"])
                for i in range(len(logprobs_data))
            ]

    return None


def check_model_availability(provider: str, model: str) -> tuple[bool, str]:
    """Check if the configured LLM provider is reachable and the asked model is available."""
    if provider in ("openai", "openrouter"):
        import openai

        try:
            api_key = os.environ.get(
                "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
            )
            if not api_key:
                error_message = f"Missing API key for {provider}. Set the {'OPENROUTER_API_KEY' if provider == 'openrouter' else 'OPENAI_API_KEY'} environment variable."
                logger.error(error_message)
                return False, error_message

            models = []
            if provider == "openai":
                models = openai.OpenAI(api_key=api_key, timeout=5.0).models.list()
            elif provider == "openrouter":
                models = openai.OpenAI(
                    api_key=api_key,
                    timeout=5.0,
                    base_url="https://openrouter.ai/api/v1",
                ).models.list()

            models = [provider_model.id for provider_model in models]
            if model not in models:
                raise ValueError(
                    f"The requested model is not available: '{model}'. Available models are {models}"
                )
            return True, f"Connection to {provider} succeeded."
        except openai.OpenAIError as e:
            error_message = f"{provider} connection failed: {e}"
            logger.error(error_message)
            return False, error_message
        except ValueError as e:
            error_message = f"{provider} connection failed: {e}"
            logger.error(error_message)
            return False, error_message

    # Ollama
    try:
        base = OLLAMA_URL.rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [model.get("model").split(":")[0] for model in resp.json()["models"]]
        if model.split(":")[0] not in models:
            raise ValueError(
                f"The requested model is not available: '{model}'. Available models are {models}"
            )
        return True, f"Connection to {provider} succeeded."
    except requests.RequestException as e:
        error_message = f"{provider} connection failed: {e}"
        logger.error(error_message)
        return False, error_message
    except ValueError as e:
        error_message = f"{provider} connection failed: {e}"
        logger.error(error_message)
        return False, error_message


def fill_json_schema(
    parsed: dict[str, Any] | None,
    schema: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fill a parsed JSON dict against a criterion schema.

    For each criterion in *schema*:
    - If missing from *parsed* or not a dict → use the full *defaults*.
    - Otherwise → merge *defaults* under the parsed entry and clamp
      *score* to **[1, 5]**, falling back to the default if the score
      is non-numeric.

    Returns a new dict guaranteed to contain every criterion from the schema.
    """
    result: dict[str, Any] = {}
    for criterion, defaults in schema.items():
        entry = (parsed or {}).get(criterion)
        if not isinstance(entry, dict):
            result[criterion] = dict(defaults)
        else:
            merged = {**defaults, **entry}
            raw_score = merged.get("score", defaults["score"])
            if not isinstance(raw_score, (int, float)):
                raw_score = defaults["score"]
            merged["score"] = max(1, min(5, round(raw_score)))
            result[criterion] = merged
    return result


def compute_weighted_score(
    evaluation: dict[str, Any], weights: dict[str, float]
) -> float:
    """Normalise 1-5 criterion scores to a [0, 1] weighted average."""
    total = 0.0
    for criterion, weight in weights.items():
        raw = evaluation.get(criterion, {}).get("score", 1)
        normalized = (raw - 1) / 4.0
        total += normalized * weight
    return round(total, 6)


__all__ = [
    "LLM",
    "LLMAnswer",
    "check_model_availability",
    "compute_weighted_score",
    "fill_json_schema",
    "get_llm",
    "get_llm_from_config",
]
