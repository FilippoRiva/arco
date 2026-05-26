from langchain_core.messages import AIMessage
import os
import time
from typing import Optional, Dict, Any, TYPE_CHECKING

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from arco.global_vars import OLLAMA_REQUEST_TIMEOUT

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


class LLMCallAccumulator(BaseCallbackHandler):
    """Accumulates wall-clock time and energy of LLM .invoke() calls via LangChain callbacks.

    Attach as a callback to a LangChain LLM object to record only the time and energy
    spent inside actual LLM API calls, excluding non-LLM work (DB queries, parquet reads,
    code execution, etc.) that may be present in the same step function.

    When cc_enabled=True, a fresh CodeCarbon EmissionsTracker is started at the beginning
    of each invoke() and stopped at the end. This avoids the pro-rating approximation that
    would be incorrect when GPU power varies significantly during inference (e.g. local
    Ollama on A40/L40S), since the tracker window covers only the actual inference window.

    Thread-safe for sequential use (one step at a time).
    """

    _save_dir: str | None = None
    _enabled: bool = False

    def __init__(self, name:str) -> None:
        super().__init__()
        self._starts: Dict[str, float | int] = {}
        self._cc_trackers: Dict[str, Any] = {}
        self.total_time: float | int = 0.0
        self._cc_output_dir : str | None = os.path.join(LLMCallAccumulator._save_dir, name) if LLMCallAccumulator._save_dir else None
        self._enabled : bool = LLMCallAccumulator._enabled
        # Accumulated energy across all invoke() calls for this step
        self.total_energy: Dict[str, float | int] = {}

        if self._cc_output_dir:
            os.makedirs(self._cc_output_dir, exist_ok=True)

    @staticmethod
    def enable(save_dir: str):
        LLMCallAccumulator._save_dir = save_dir
        LLMCallAccumulator._enabled = True

    def _start_cc_tracker(self, key: str) -> None:
        if not self._enabled:
            return
        try:
            tracker = EmissionsTracker(  # type: ignore[call-arg]
                project_name="llm_invoke",
                output_dir=self._cc_output_dir,
                save_to_file=False,
                measure_power_secs=1,
                log_level="error",
                allow_multiple_runs=True,
            )
            tracker.start()
            self._cc_trackers[key] = tracker
        except Exception as _e:
            pass
            # print(f"[CodeCarbon] per-invoke tracker start failed: {_e}")

    def _stop_cc_tracker(self, key: str) -> None:
        tracker = self._cc_trackers.pop(key, None)
        if tracker is None:
            return
        try:
            tracker.stop()
            # tracker.stop() returns a float (CO2 kg), not EmissionsData.
            # The full breakdown is in final_emissions_data, same as the original code.
            _ed = getattr(tracker, "final_emissions_data", None)
            if _ed is not None:
                self.total_energy["energy_consumed_kwh"] += getattr(_ed, "energy_consumed", 0.0) or 0.0
                self.total_energy["cpu_energy_kwh"] += getattr(_ed, "cpu_energy", 0.0) or 0.0
                self.total_energy["gpu_energy_kwh"] += getattr(_ed, "gpu_energy", 0.0) or 0.0
                self.total_energy["ram_energy_kwh"] += getattr(_ed, "ram_energy", 0.0) or 0.0
                self.total_energy["emissions_kg_co2"] += getattr(_ed, "emissions", 0.0) or 0.0
        except Exception as _e:
            pass
            # print(f"[CodeCarbon] per-invoke tracker stop failed: {_e}")

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        key = str(run_id)
        self._starts[key] = time.perf_counter()
        self._start_cc_tracker(key)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        key = str(run_id)
        if key in self._starts:
            self.total_time += time.perf_counter() - self._starts.pop(key)
        self._stop_cc_tracker(key)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        # Count errored calls too — the HTTP round-trip still happened.
        key = str(run_id)
        if key in self._starts:
            self.total_time += time.perf_counter() - self._starts.pop(key)
        self._stop_cc_tracker(key)

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
