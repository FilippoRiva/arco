"""Energy and timing tracking via LangChain callbacks and CodeCarbon.

This module provides :class:`LLMCallAccumulator`, a LangChain callback
handler that measures wall-clock time and (optionally) energy consumption
for each LLM ``.invoke()`` call.  :func:`initialize_tracking` is called
once per workflow run to enable CodeCarbon integration.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codecarbon import OfflineEmissionsTracker

    from .config import Config

import logging
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_count(values: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        count = _count(values.get(key))
        if count is not None:
            return count
    return None


def _add_counts(first: int | None, second: int | None) -> int | None:
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported for one or more model invocations."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=_add_counts(self.input_tokens, other.input_tokens),
            output_tokens=_add_counts(self.output_tokens, other.output_tokens),
            total_tokens=_add_counts(self.total_tokens, other.total_tokens),
            cache_creation_tokens=_add_counts(
                self.cache_creation_tokens, other.cache_creation_tokens
            ),
            cache_read_tokens=_add_counts(
                self.cache_read_tokens, other.cache_read_tokens
            ),
            reasoning_tokens=_add_counts(self.reasoning_tokens, other.reasoning_tokens),
        )

    @property
    def has_data(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.cache_creation_tokens,
                self.cache_read_tokens,
                self.reasoning_tokens,
            )
        )

    @classmethod
    def from_usage_metadata(cls, value: Any) -> TokenUsage | None:
        """Read LangChain AIMessage ``usage_metadata`` when available."""
        usage = _mapping(value)
        if not usage:
            return None

        input_details = _mapping(usage.get("input_token_details"))
        output_details = _mapping(usage.get("output_token_details"))
        input_tokens = _first_count(usage, "input_tokens", "prompt_tokens")
        output_tokens = _first_count(usage, "output_tokens", "completion_tokens")
        total_tokens = _first_count(usage, "total_tokens")
        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens
        token_usage = cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_creation_tokens=_first_count(
                input_details, "cache_creation", "cache_creation_tokens"
            ),
            cache_read_tokens=_first_count(
                input_details, "cache_read", "cache_read_tokens", "cached_tokens"
            ),
            reasoning_tokens=_first_count(
                output_details, "reasoning", "reasoning_tokens"
            ),
        )
        return token_usage if token_usage.has_data else None

    @classmethod
    def from_llm_result(cls, result: Any) -> TokenUsage | None:
        """Extract usage from an AIMessage or a LangChain ``LLMResult``."""
        direct_usage = cls.from_usage_metadata(getattr(result, "usage_metadata", None))
        if direct_usage is not None:
            return direct_usage

        generation_usages = []
        for generation_group in getattr(result, "generations", ()) or ():
            for generation in generation_group:
                message_usage = cls.from_usage_metadata(
                    getattr(
                        getattr(generation, "message", None), "usage_metadata", None
                    )
                )
                if message_usage is not None:
                    generation_usages.append(message_usage)

        if generation_usages:
            total = cls()
            for usage in generation_usages:
                total += usage
            return total

        llm_output = _mapping(getattr(result, "llm_output", None))
        provider_usage = _mapping(llm_output.get("token_usage"))
        if not provider_usage:
            return None

        input_details = _mapping(
            provider_usage.get("input_token_details")
            or provider_usage.get("prompt_tokens_details")
        )
        output_details = _mapping(
            provider_usage.get("output_token_details")
            or provider_usage.get("completion_tokens_details")
        )
        input_tokens = _first_count(provider_usage, "input_tokens", "prompt_tokens")
        output_tokens = _first_count(
            provider_usage, "output_tokens", "completion_tokens"
        )
        total_tokens = _first_count(provider_usage, "total_tokens")
        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens
        token_usage = cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_creation_tokens=_first_count(
                input_details, "cache_creation", "cache_creation_tokens"
            ),
            cache_read_tokens=_first_count(
                input_details, "cache_read", "cache_read_tokens", "cached_tokens"
            ),
            reasoning_tokens=_first_count(
                output_details, "reasoning", "reasoning_tokens"
            ),
        )
        return token_usage if token_usage.has_data else None


def reasoning_value_to_text(value: Any) -> str:
    """Normalize provider-specific reasoning values to plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            text for item in value for text in [reasoning_value_to_text(item)] if text
        )
    if isinstance(value, Mapping):
        for key in ("text", "reasoning", "reasoning_content", "summary"):
            if key in value:
                text = reasoning_value_to_text(value[key])
                if text:
                    return text
    return ""


def extract_reasoning(response: Any) -> str:
    """Extract provider-native reasoning text from an AI message."""
    parts: list[str] = []
    for block in getattr(response, "content_blocks", []) or []:
        if isinstance(block, Mapping) and block.get("type") == "reasoning":
            text = reasoning_value_to_text(
                block.get("reasoning") or block.get("summary")
            )
            if text:
                parts.append(text)

    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning"):
        text = reasoning_value_to_text(additional_kwargs.get(key))
        if text and text not in parts:
            parts.append(text)

    return "\n".join(parts)


def _token_logprob_pairs(values: Any) -> list[tuple[str, float | int]]:
    pairs = []
    for value in values or []:
        if not isinstance(value, Mapping):
            continue
        token = value.get("token")
        logprob = value.get("logprob")
        if (
            isinstance(token, str)
            and isinstance(logprob, int | float)
            and not isinstance(logprob, bool)
        ):
            pairs.append((token, logprob))
    return pairs


def extract_logprobs(message: Any) -> list[tuple[str, float | int]]:
    """Extract token/log-probability pairs from common provider metadata."""
    metadata = getattr(message, "response_metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return []
    logprobs_data = metadata.get("logprobs")
    if logprobs_data is None:
        return []

    model_name = str(metadata.get("model_name", metadata.get("model", ""))).lower()
    if isinstance(logprobs_data, Mapping) and "content" in logprobs_data:
        token_logprobs = _token_logprob_pairs(logprobs_data.get("content"))

        if "deepseek" in model_name:
            tokens = [token for token, _ in token_logprobs]
            if "</think>" in tokens:
                token_logprobs = token_logprobs[tokens.index("</think>") + 1 :]
            tokens = [token for token, _ in token_logprobs]
            if "<｜end▁of▁sentence｜>" in tokens:
                token_logprobs = token_logprobs[: tokens.index("<｜end▁of▁sentence｜>")]
        return token_logprobs

    if isinstance(logprobs_data, list):
        token_logprobs = _token_logprob_pairs(logprobs_data)
        if "gemma4" in model_name:
            tokens = [token for token, _ in token_logprobs]
            if "<channel|>" in tokens:
                token_logprobs = token_logprobs[tokens.index("<channel|>") + 1 :]
        return token_logprobs

    return []


def _configure_codecarbon_logger() -> None:
    """Route CodeCarbon records to ARCO's run log instead of the console."""
    codecarbon_logger = logging.getLogger("codecarbon")
    root_file_handler = next(
        (
            handler
            for handler in reversed(logging.getLogger().handlers)
            if isinstance(handler, logging.FileHandler)
        ),
        None,
    )
    if root_file_handler is None:
        # Outside an ARCO CLI run, keep CodeCarbon quiet unless it reports errors.
        codecarbon_logger.setLevel(logging.ERROR)
        return

    # CodeCarbon installs its own StreamHandler and sets propagate=False. Attach
    # the active ARCO file handler directly and discard CodeCarbon's console one.
    codecarbon_logger.handlers[:] = [root_file_handler]
    codecarbon_logger.setLevel(logging.INFO)
    codecarbon_logger.propagate = False


def initialize_tracking(config: Config) -> None:
    """Enable CodeCarbon energy tracking for a workflow run.

    Does nothing if ``config.enable_codecarbon`` is ``False``. When enabled,
    turns on tracking for subsequently created :class:`LLMCallAccumulator`
    instances.

    :param config: The workflow configuration.
    """
    if not config.enable_codecarbon:
        return
    LLMCallAccumulator.enable()
    logger.info("Initialized codecarbon tracking")


class LLMCallAccumulator(BaseCallbackHandler):
    """Accumulates wall-clock time and energy of LLM ``.invoke()`` calls.

    Attach as a callback to a LangChain LLM for per-call timing. When
    CodeCarbon is enabled, the same accumulator also tracks the complete
    agent step, including DB queries, parquet reads, code execution, and
    other non-LLM work.

    When CodeCarbon is enabled, one offline tracker is started for the
    accumulator's complete agent step and stopped by :meth:`finish`,
    collecting CPU, GPU, and RAM energy plus CO2 emissions.

    Thread-safe for sequential use (one step at a time).

    :ivar total_time: Cumulative wall-clock seconds spent in LLM calls;
        CodeCarbon energy covers the complete agent step.
    :ivar energy_dict: Cumulative energy metrics dict with keys
        ``energy_consumed_kwh``, ``cpu_energy_kwh``, ``gpu_energy_kwh``,
        ``ram_energy_kwh``, and ``emissions_kg_co2``.
    :ivar token_usage: Token counts accumulated across all LLM calls.
    :ivar last_logprobs: Log probabilities from the latest response.
    :ivar last_reasoning: Reasoning text from the latest response, if reported.
    """

    _enabled: bool = False

    def __init__(self, name: str):
        """Create an accumulator for a named agent step.

        :param name: The agent type name (used for the CodeCarbon subdirectory).
        """
        super().__init__()
        self._starts: dict[str, float | int] = {}
        self._cc_tracker: OfflineEmissionsTracker | None = None
        self.total_time: float | int = 0.0
        self._enabled: bool = LLMCallAccumulator._enabled
        self.energy_dict: dict[str, float | int] = defaultdict(float)
        self.token_usage = TokenUsage()
        self.last_logprobs: list[tuple[str, float | int]] = []
        self.last_reasoning: str | None = None

    @staticmethod
    def enable() -> None:
        """Globally enable CodeCarbon tracking for all new accumulators."""
        LLMCallAccumulator._enabled = True

    def _start_cc_tracker(self) -> None:
        if not self._enabled or self._cc_tracker is not None:
            return
        from codecarbon import OfflineEmissionsTracker

        _configure_codecarbon_logger()
        # Milan is represented by Italy/Lombardy. Offline tracking avoids
        # CodeCarbon's repeated cloud/geolocation network lookups.
        try:
            self._cc_tracker = OfflineEmissionsTracker(  # type: ignore[call-arg]
                project_name="llm_invoke",
                country_iso_code="ITA",
                region="Lombardy",
                save_to_file=False,
                measure_power_secs=1,
                log_level="error",
                allow_multiple_runs=True,
            )
        finally:
            # CodeCarbon reconfigures its logger inside the tracker constructor.
            _configure_codecarbon_logger()
        self._cc_tracker.start()

    def start(self) -> None:
        """Start agent-level CodeCarbon tracking, including non-LLM work."""
        self._start_cc_tracker()

    def finish(self) -> None:
        """Stop the agent-level tracker and collect its accumulated energy."""
        emission_tracker = self._cc_tracker
        if emission_tracker is None:
            return
        self._cc_tracker = None
        emission_tracker.stop()
        emission_data = getattr(emission_tracker, "final_emissions_data", None)
        if emission_data is None:
            return
        self.energy_dict["energy_consumed_kwh"] += (
            getattr(emission_data, "energy_consumed", 0.0) or 0.0
        )
        self.energy_dict["cpu_energy_kwh"] += (
            getattr(emission_data, "cpu_energy", 0.0) or 0.0
        )
        self.energy_dict["gpu_energy_kwh"] += (
            getattr(emission_data, "gpu_energy", 0.0) or 0.0
        )
        self.energy_dict["ram_energy_kwh"] += (
            getattr(emission_data, "ram_energy", 0.0) or 0.0
        )
        self.energy_dict["emissions_kg_co2"] += (
            getattr(emission_data, "emissions", 0.0) or 0.0
        )

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        """LangChain callback: start timing and (optionally) CodeCarbon tracking."""
        key = str(run_id)
        self._starts[key] = time.perf_counter()
        self._start_cc_tracker()

    def reset_response_metadata(self) -> None:
        """Clear response details before generating a new candidate answer."""
        self.last_logprobs = []
        self.last_reasoning = None

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        """Accumulate timing, usage, and metadata for each LLM call."""
        key = str(run_id)
        if key in self._starts:
            self.total_time += time.perf_counter() - self._starts.pop(key)
        token_usage = TokenUsage.from_llm_result(response)
        if token_usage is not None:
            self.token_usage += token_usage

        self.reset_response_metadata()
        messages = []
        if hasattr(response, "response_metadata"):
            messages.append(response)
        else:
            for generation_group in getattr(response, "generations", ()) or ():
                messages.extend(
                    message
                    for generation in generation_group
                    if (message := getattr(generation, "message", None)) is not None
                )
        if messages:
            # LangChain's invoke() returns the first generation in a result.
            last_message = messages[0]
            self.last_logprobs = extract_logprobs(last_message)
            self.last_reasoning = extract_reasoning(last_message) or None

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self.on_llm_end(response=error, run_id=run_id, **kwargs)


__all__ = [
    "LLMCallAccumulator",
    "TokenUsage",
    "extract_logprobs",
    "extract_reasoning",
    "initialize_tracking",
    "reasoning_value_to_text",
]
