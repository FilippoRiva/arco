"""LLM abstraction layer: wrappers, factories, and utilities.

This module provides :class:`LLMAnswer` (pre-extracted response with
logprobs), :class:`LLM` (thin wrapper with iterative refinement), factory
functions for creating LLM instances, and utilities for JSON parsing
and response extraction used across evaluators and agents.
"""

import json
import logging
import os
import re
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .tracking import LLMCallAccumulator

# Global parameters
OLLAMA_REQUEST_TIMEOUT: int = 600
OLLAMA_URL: str = "http://localhost:11434"

DEFAULT_LLM_ACC = LLMCallAccumulator("None")

if TYPE_CHECKING:
    from .agent_config import AgentConfig


logger = logging.getLogger(__name__)


def _message_text(response: AIMessage) -> str:
    """Return only text blocks from a provider-native message response."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def _reasoning_value_to_text(value: Any) -> str:
    """Normalise provider-specific reasoning fields to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            text for item in value for text in [_reasoning_value_to_text(item)] if text
        )
    if isinstance(value, dict):
        for key in ("text", "reasoning", "reasoning_content", "summary"):
            if key in value:
                text = _reasoning_value_to_text(value[key])
                if text:
                    return text
    return ""


def _extract_reasoning(response: AIMessage) -> str:
    """Extract reasoning from OpenAI, Ollama, and OpenRouter message shapes."""
    parts: list[str] = []
    for block in getattr(response, "content_blocks", []) or []:
        if isinstance(block, dict) and block.get("type") == "reasoning":
            text = _reasoning_value_to_text(
                block.get("reasoning") or block.get("summary")
            )
            if text:
                parts.append(text)

    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning"):
        text = _reasoning_value_to_text(additional_kwargs.get(key))
        if text and text not in parts:
            parts.append(text)

    return "\n".join(parts)


def _log_raw_response(response: AIMessage) -> None:
    """Log all provider response fields useful when debugging reasoning."""
    logger.debug("RAW CONTENT ... %r", getattr(response, "content", None))
    logger.debug("RAW CONTENT BLOCKS ... %r", getattr(response, "content_blocks", None))
    logger.debug(
        "RAW ADDITIONAL KWARGS ... %r",
        getattr(response, "additional_kwargs", None),
    )
    logger.debug(
        "RAW RESPONSE METADATA ... %r",
        getattr(response, "response_metadata", None),
    )


class LLMAnswer:
    """Pre-extracted LLM response — no manual content/logprobs wrangling.

    Wraps a raw LangChain :class:`AIMessage` (or any object with
    ``.content``) and provides convenience methods for extracting
    fenced code blocks, JSON, Python code, and SQL.

    :ivar text: The raw response text.
    :ivar logprobs: Token-level log probabilities as ``(token, logprob)``
        tuples, or ``None`` if not available.
    """

    def __init__(self, response: AIMessage):
        self.text: str = _message_text(response)
        self.logprobs: list[tuple[str, float | int]] = _extract_logprobs(response)
        self.reasoning: str = _extract_reasoning(response)
        logger.debug("Reasoning output: %s", self.reasoning)

    def extract_fenced_content(self) -> str:
        """Extract content from a Markdown fenced code block.

        Searches for the first triple-backtick fence and returns its
        contents.  If no fence is found, returns the raw text.

        :returns: The fenced content with surrounding whitespace stripped.
        """
        fence_re = re.compile(r"```[^\n]*\n?(.*?)\n?```", re.DOTALL)

        match = fence_re.search(self.text)
        content = match.group(1).strip() if match else self.text.strip()
        return content

    def extract_json(self) -> dict:
        """Extract and parse the first JSON object from the response.

        :returns: The parsed dict, or an empty dict on failure.
        """
        content = self.extract_fenced_content()
        try:
            return json.loads(content)
        except JSONDecodeError:
            return {}

    def extract_json_list(self) -> list:
        """Extract and parse the first JSON array from the response.

        :returns: The parsed list, or an empty list on failure.
        """
        content = self.extract_fenced_content()
        try:
            json_list = json.loads(content)
            if not isinstance(json_list, list):
                raise TypeError("Parsed json didn't produce a list")
            return json_list
        except JSONDecodeError, TypeError:
            return []

    def extract_python(self) -> str:
        """Extract Python code from a Markdown fenced block.

        :returns: The fenced content (same as :meth:`extract_fenced_content`).
        """
        return self.extract_fenced_content()

    def extract_sql(self) -> str:
        """Extract SQL code from a Markdown fenced block.

        :returns: The fenced content (same as :meth:`extract_fenced_content`).
        """
        return self.extract_fenced_content()


class _OpenRouterChatOpenAI(ChatOpenAI):
    """ChatOpenAI adapter that preserves OpenRouter reasoning fields.

    OpenRouter returns reasoning on the assistant message using fields that
    the standard OpenAI message converter intentionally drops. Preserve those
    fields as ``reasoning_content`` so the common extractor can consume them.
    """

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(exclude_none=False, warnings=False)
        )

        for choice, generation in zip(
            response_dict.get("choices", []), result.generations, strict=False
        ):
            raw_message = choice.get("message", {})
            reasoning = (
                raw_message.get("reasoning")
                or raw_message.get("reasoning_content")
                or raw_message.get("reasoning_details")
            )
            if reasoning:
                generation.message.additional_kwargs["reasoning_content"] = (
                    _reasoning_value_to_text(reasoning)
                )
                if raw_message.get("reasoning_details"):
                    generation.message.additional_kwargs["reasoning_details"] = (
                        raw_message["reasoning_details"]
                    )

        return result


class LLM:
    """Thin wrapper around a LangChain chat model with iterative refinement.

    Wraps a :class:`BaseChatModel` and can append an iterative refinement
    instruction to the previous response. This is not provider reasoning
    generation or a reasoning trace.

    :ivar iterative_refinement_enabled: If ``True``, :meth:`invoke` applies
        iterative refinement.
    :ivar last_answer: The most recent :class:`LLMAnswer` produced.
    :ivar execution_error: Error string from the last execution, used as
        refinement feedback.
    """

    _ITERATIVE_REFINEMENT_SUFFIX = """
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

    _ITERATIVE_REFINEMENT_ERROR_SUFFIX = """
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
        self.iterative_refinement_enabled: bool = False
        self.last_answer: LLMAnswer | None = None
        self.execution_error: str | None = None

    def _invoke_raw(self, prompt: str) -> AIMessage:
        """Invoke the model, retrying without logprobs if unsupported."""
        try:
            return self._chat_model.invoke(prompt)
        except Exception as exc:
            error_text = str(exc).lower()
            if (
                "logprobs" not in error_text
                or getattr(self._chat_model, "logprobs", None) is not True
            ):
                raise

            logger.warning(
                "Model %s does not support logprobs; retrying without it: %s",
                getattr(self._chat_model, "model_name", "unknown"),
                exc,
            )
            self._chat_model = self._chat_model.model_copy(update={"logprobs": None})
            return self._chat_model.invoke(prompt)

    def invoke(self, prompt: str) -> LLMAnswer:
        """Send a prompt to the LLM and return the response.

        If :attr:`iterative_refinement_enabled` is ``True``, the prompt is extended with
        a refinement suffix based on the previous answer and any
        execution error.

        :param prompt: The prompt string.
        :returns: An :class:`LLMAnswer` with the response text and logprobs.
        """
        if self.iterative_refinement_enabled:
            answer = self._iterative_refinement_invoke(prompt, self.execution_error)
        else:
            logger.debug(f"Invoking LLM with prompt : {prompt}")
            response = self._invoke_raw(prompt)
            _log_raw_response(response)
            answer = LLMAnswer(response)
        logger.debug(f"Answer text : {answer.text}")
        self.last_answer = answer
        return answer

    def _iterative_refinement_invoke(
        self, prompt: str, execution_error: str | None
    ) -> LLMAnswer:
        if self.last_answer is None:
            raise ValueError(
                "Iterative refinement requires a previous answer"
            )
        if execution_error:
            suffix = self._ITERATIVE_REFINEMENT_ERROR_SUFFIX.format(
                previous_response=self.last_answer.text,
                execution_error=execution_error,
            )
        else:
            suffix = self._ITERATIVE_REFINEMENT_SUFFIX.format(
                previous_response=self.last_answer.text,
            )
        prompt = prompt + suffix
        logger.debug(f"Invoking iterative-refinement LLM with prompt : {prompt}")
        response = self._invoke_raw(prompt)
        _log_raw_response(response)
        return LLMAnswer(response)


def get_llm_from_config(agent_config: AgentConfig, llm_acc: LLMCallAccumulator) -> LLM:
    """Create an :class:`LLM` from an :class:`AgentConfig`, using the first candidate parameters.

    :param agent_config: The agent's configuration (used for provider, model, temperature, etc.).
    :param llm_acc: The callback accumulator for timing and energy tracking.
    :returns: A configured :class:`LLM` instance.
    """
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
        enable_reasoning=bool(agent_config.enable_reasoning),
        reasoning_effort=agent_config.reasoning_effort,
        reasoning_summary=agent_config.reasoning_summary,
        reasoning_max_tokens=agent_config.reasoning_max_tokens,
        verbosity=agent_config.verbosity,
        enable_logprobs=bool(agent_config.enable_logprobs),
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
    enable_reasoning: bool = False,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    reasoning_max_tokens: int | None = None,
    verbosity: str | None = None,
    enable_logprobs: bool = True,
) -> LLM:
    """Factory to create an :class:`LLM` instance with specific parameters.

    Creates a new LLM instance instead of mutating a global instance,
    which allows per-step parameter customisation.

    :param temperature: Sampling temperature.
    :param max_tokens: Maximum tokens for generation.
    :param top_p: Top-p (nucleus) sampling parameter.
    :param top_k: Top-k sampling parameter (skipped for OpenAI).
    :param num_beams: Beam search width, 1 = greedy/disabled (skipped for OpenAI).
    :param no_repeat_ngram_size: Prevent repeating n-grams (skipped for OpenAI).
    :param streaming: Whether to stream the response tokens.
    :param provider: The LLM provider (``'openai'``, ``'openrouter'``, or ``'ollama'``).
    :param model: The specific model ID to instantiate.
    :param llm_accumulator: Callback accumulator for timing and energy tracking.
    :param openrouter_url: Base URL for the OpenRouter API.
    :param enable_reasoning: Request provider-supported reasoning summaries or
        thinking output. Unsupported models may ignore or reject this option.
    :param reasoning_effort: Provider-specific reasoning effort, such as
        ``low``, ``medium``, or ``high``.
    :param reasoning_summary: OpenAI Responses reasoning summary mode.
    :param reasoning_max_tokens: Direct reasoning-token budget where supported.
    :param verbosity: Visible response verbosity where supported.
    :param enable_logprobs: Request token log probabilities where supported.
        This is automatically omitted for the OpenAI Responses API.
    :returns: A configured :class:`LLM` instance.
    :raises ValueError: If the provider is ``'openrouter'`` but the
        ``OPENROUTER_API_KEY`` environment variable is not set.
    """
    if provider.lower() == "openai":
        openai_kwargs = {
            "model": model,
            "streaming": streaming,
            "callbacks": [llm_accumulator],
            # ChatOpenAI maps max_tokens to max_completion_tokens for Chat
            # Completions and to max_output_tokens for Responses API.
            "max_tokens": max_tokens,
        }
        # These sampling options are valid for normal Chat Completions. Do
        # not send them for reasoning Responses calls: reasoning models may
        # reject temperature/top_p just as Responses rejects logprobs.
        if not enable_reasoning:
            if temperature is not None:
                openai_kwargs["temperature"] = temperature
            if top_p is not None:
                openai_kwargs["top_p"] = top_p
            if enable_logprobs:
                openai_kwargs["logprobs"] = True
        if enable_reasoning:
            if enable_logprobs:
                logger.debug(
                    "Disabling logprobs for %s: OpenAI Responses API does not support it",
                    model,
                )
            if temperature is not None or top_p is not None:
                logger.debug(
                    "Disabling temperature/top_p for %s: reasoning Responses call",
                    model,
                )
            # OpenAI reasoning summaries are available through the Responses
            # API. Raw private reasoning is not exposed by OpenAI.
            openai_kwargs.update(
                {
                    "use_responses_api": True,
                    "output_version": "responses/v1",
                    "reasoning": {
                        "effort": reasoning_effort or "medium",
                        "summary": reasoning_summary or "auto",
                    },
                    "verbosity": verbosity,
                }
            )
        chat_model = ChatOpenAI(**openai_kwargs)
    elif provider.lower() == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter requires an API key: pass openrouter_api_key or set the OPENROUTER_API_KEY environment variable."
            )
        # Optional parameters are not uniformly supported by every routed
        # provider. Let OpenRouter choose a compatible endpoint instead of
        # rejecting the whole request when one provider lacks a parameter.
        openrouter_extra_body = {
            "provider": {
                "require_parameters": False,
            }
        }
        if enable_reasoning:
            if enable_logprobs:
                logger.debug(
                    "Disabling logprobs for OpenRouter reasoning request on %s",
                    model,
                )
            if verbosity is not None:
                logger.debug(
                    "Ignoring verbosity for OpenRouter request on %s",
                    model,
                )
            # OpenRouter accepts either an effort level or a direct reasoning
            # token budget, but not both in the same request.
            reasoning_config = {"enabled": True}
            if reasoning_max_tokens is not None:
                reasoning_config["max_tokens"] = reasoning_max_tokens
            else:
                reasoning_config["effort"] = reasoning_effort or "medium"
            openrouter_extra_body["reasoning"] = reasoning_config
        chat_model = _OpenRouterChatOpenAI(
            model=model,
            api_key=SecretStr(api_key),
            base_url=openrouter_url,
            temperature=temperature,
            max_tokens=max_tokens,
            # ChatOpenAI drops OpenRouter reasoning fields from streaming
            # deltas. Use one non-streamed response when reasoning is enabled
            # so _create_chat_result can preserve the final reasoning field.
            streaming=streaming and not enable_reasoning,
            callbacks=[llm_accumulator],
            # Reasoning endpoints commonly do not expose token logprobs.
            logprobs=enable_logprobs and not enable_reasoning,
            extra_body=openrouter_extra_body,
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
            "logprobs": enable_logprobs,
            "reasoning": (
                reasoning_effort
                if enable_reasoning and reasoning_effort
                else enable_reasoning
            ),
        }

        if top_k is not None:
            kwargs["top_k"] = top_k
        if num_beams is not None and num_beams > 1:
            kwargs["num_beams"] = num_beams
        if no_repeat_ngram_size is not None:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        chat_model = ChatOllama(**kwargs)
    return LLM(base_chat_model=chat_model)


def _extract_logprobs(message: AIMessage) -> list[tuple[str, float | int]]:
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

    return []


def check_model_availability(provider: str, model: str) -> tuple[bool, str]:
    """Check whether the configured LLM provider is reachable and the model exists.

    For OpenAI / OpenRouter, queries the models API.  For Ollama, queries
    the local ``/api/tags`` endpoint.

    :param provider: The provider name (``'openai'``, ``'openrouter'``, or ``'ollama'``).
    :param model: The model ID to look up.
    :returns: A tuple ``(available, message)`` where *available* is
        ``True`` if the model was found and *message* describes the result.
    """
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

    :param parsed: The parsed JSON dict (may be ``None``).
    :param schema: A dict mapping criterion names to their default value dicts.
    :returns: A new dict guaranteed to contain every criterion from the schema.
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
    """Normalize 1-5 criterion scores to a ``[0, 1]`` weighted average.

    :param evaluation: A dict mapping criterion names to dicts with a
        ``"score"`` key (value in ``1-5``).
    :param weights: A dict mapping criterion names to their weight.
    :returns: The weighted average, normalised to ``[0, 1]``.
    """
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
