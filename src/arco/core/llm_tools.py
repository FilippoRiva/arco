import os
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from arco.core.tracking import LLMCallAccumulator

# Global parameters
OLLAMA_REQUEST_TIMEOUT: int = 600
OLLAMA_URL: str = "http://localhost:11434"

DEFAULT_LLM_ACC = LLMCallAccumulator("None")

if TYPE_CHECKING:
    from arco.core import AgentConfig


class CoTRefiner:
    """Transparent LLM wrapper that appends the previous iteration's response for iterative CoT refinement.

    Every call to invoke() receives the original prompt augmented with a
    refinement block containing the previous iteration's output.  The wrapper
    delegates all other attribute accesses to the underlying LLM so that the
    core step functions need no changes.
    """

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

    def __init__(
        self, base_llm, previous_response: str, execution_error: str | None = None
    ) -> None:
        self._llm = base_llm
        self._previous_response = previous_response
        self._execution_error = execution_error

    def invoke(self, prompt):
        if self._execution_error:
            suffix = self._ERROR_SUFFIX.format(
                previous_response=self._previous_response,
                execution_error=self._execution_error,
            )
        else:
            suffix = self._REFINEMENT_SUFFIX.format(
                previous_response=self._previous_response,
            )
        return self._llm.invoke(prompt + suffix)

    def __getattr__(self, name):
        return getattr(self._llm, name)


def get_llm_from_config(
    agent_config: AgentConfig, llm_acc: LLMCallAccumulator
) -> BaseChatModel:
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
) -> BaseChatModel:
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
        return ChatOpenAI(
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
        return ChatOpenAI(
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
        return ChatOllama(**kwargs)


def extract_logprobs(message: AIMessage) -> list[tuple[str, float | int]] | None:
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
