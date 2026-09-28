"""Energy and timing tracking via LangChain callbacks and CodeCarbon.

This module provides :class:`LLMCallAccumulator`, a LangChain callback
handler that measures wall-clock time and (optionally) energy consumption
for each LLM ``.invoke()`` call.  :func:`initialize_tracking` is called
once per workflow run to enable CodeCarbon integration.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codecarbon import OfflineEmissionsTracker

    from .config import Config

import logging
from collections import defaultdict

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


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

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        """LangChain callback: stop timing for this LLM call."""
        key = str(run_id)
        if key in self._starts:
            self.total_time += time.perf_counter() - self._starts.pop(key)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self.on_llm_end(response=error, run_id=run_id, **kwargs)


__all__ = ["LLMCallAccumulator", "initialize_tracking"]
