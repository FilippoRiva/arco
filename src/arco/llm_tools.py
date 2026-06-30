from langchain_core.messages import AIMessage
from typing import Optional, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from arco.global_vars import OLLAMA_REQUEST_TIMEOUT
from arco.tracking import LLMCallAccumulator

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

    def __init__(self, base_llm, previous_response: str, execution_error: str | None = None ) -> None:
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


def get_llm_from_config(agent_config: AgentConfig, llm_acc: LLMCallAccumulator) -> BaseChatModel:
    temp, top_p, top_k = agent_config.get_candidate_params()[0]

    if agent_config.provider is None or agent_config.model is None:
        raise Exception("Agent's provider and model should be specified")

    return get_llm(
        provider=agent_config.provider,
        model=agent_config.model,
        max_tokens=agent_config.max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        num_beams=agent_config.num_beams,
        no_repeat_ngram_size=agent_config.no_repeat_ngram_size,
        ollama_url=agent_config.ollama_url,
        llm_accumulator=llm_acc,
    )


def get_llm(
        provider: str = 'openai',
        model: str = 'gpt-4o-mini',
        streaming=False,
        max_tokens: int = 2000,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_beams: int | None = None,
        no_repeat_ngram_size: Optional[int] = None,
        llm_accumulator: LLMCallAccumulator = LLMCallAccumulator("None"),
        ollama_url: Optional[str] = None,
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
        ollama_url: Base URL for the Ollama API, if using the Ollama provider.

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
            logprobs=True
        )
    else:
        kwargs = dict(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            streaming=streaming,
            base_url=ollama_url,
            top_p=top_p,
            client_kwargs={"timeout": OLLAMA_REQUEST_TIMEOUT},
            callbacks=[llm_accumulator],
            logprobs=True
        )
        if top_k is not None:
            kwargs["top_k"] = top_k
        if num_beams is not None and num_beams > 1:
            kwargs["num_beams"] = num_beams
        if no_repeat_ngram_size is not None:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        return ChatOllama(**kwargs)

def extract_logprobs(message: AIMessage) -> list[float | int] | None:
    logprobs = None

    metadata = message.response_metadata
    if "logprobs" in metadata and metadata["logprobs"] is not None:
        content_logprobs = metadata['logprobs'].get("content", [])
        logprobs = [
            token_info.get("logprob") for token_info in content_logprobs if "logprob" in token_info
        ]

    return logprobs
