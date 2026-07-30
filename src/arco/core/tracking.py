"""Energy and timing tracking via LangChain callbacks and CodeCarbon.

This module provides :class:`LLMCallAccumulator`, a LangChain callback
handler that measures wall-clock time and (optionally) energy consumption
for each LLM ``.invoke()`` call.  :func:`initialize_tracking` is called
once per workflow run to enable CodeCarbon integration.
"""

import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

import logging
from collections import defaultdict

from codecarbon import EmissionsTracker
from langchain_core.callbacks import BaseCallbackHandler

logging.getLogger("codecarbon").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def initialize_tracking(config: Config) -> None:
    """Enable CodeCarbon energy tracking for a workflow run.

    Does nothing if ``config.enable_codecarbon`` is ``False``.
    Creates the CodeCarbon output directory and enables tracking on
    all subsequently created :class:`LLMCallAccumulator` instances.

    :param config: The workflow configuration.
    """
    if not config.enable_codecarbon:
        return
    codecarbon_dir = os.path.join(config.save_dir or "./output", "codecarbon")
    os.makedirs(codecarbon_dir, exist_ok=True)
    LLMCallAccumulator.enable(codecarbon_dir)
    logger.info("Initialized codecarbon tracking")


class LLMCallAccumulator(BaseCallbackHandler):
    """Accumulates wall-clock time and energy of LLM ``.invoke()`` calls.

    Attach as a callback to a LangChain LLM to measure only the time
    and energy spent inside actual LLM calls, excluding non-LLM
    work (DB queries, parquet reads, code execution, etc.) that may
    be present in the same step function.

    When CodeCarbon is enabled, a fresh :class:`EmissionsTracker` is
    started at the beginning of each ``invoke()`` and stopped at the
    end, collecting CPU, GPU, and RAM energy plus CO2 emissions.

    Thread-safe for sequential use (one step at a time).

    :ivar total_time: Cumulative wall-clock seconds spent in LLM calls.
    :ivar energy_dict: Cumulative energy metrics dict with keys
        ``energy_consumed_kwh``, ``cpu_energy_kwh``, ``gpu_energy_kwh``,
        ``ram_energy_kwh``, and ``emissions_kg_co2``.
    """

    _save_dir: str | None = None
    _enabled: bool = False

    def __init__(self, name: str):
        """Create an accumulator for a named agent step.

        :param name: The agent type name (used for the CodeCarbon subdirectory).
        """
        super().__init__()
        self._starts: dict[str, float | int] = {}
        self._cc_trackers: dict[str, Any] = {}
        self.total_time: float | int = 0.0
        self._cc_output_dir: str | None = (
            os.path.join(LLMCallAccumulator._save_dir, name)
            if LLMCallAccumulator._save_dir
            else None
        )
        self._enabled: bool = LLMCallAccumulator._enabled
        self.energy_dict: dict[str, float | int] = defaultdict(float)

        if self._cc_output_dir:
            os.makedirs(self._cc_output_dir, exist_ok=True)

    @staticmethod
    def enable(save_dir: str) -> None:
        """Globally enable CodeCarbon tracking for all new accumulators.

        :param save_dir: Base directory for CodeCarbon output files.
        """
        LLMCallAccumulator._save_dir = save_dir
        LLMCallAccumulator._enabled = True

    def _start_cc_tracker(self, key: str) -> None:
        if not self._enabled:
            return
        emission_tracker = EmissionsTracker(  # type: ignore[call-arg]
            project_name="llm_invoke",
            output_dir=self._cc_output_dir,
            save_to_file=False,
            measure_power_secs=1,
            log_level="error",
            allow_multiple_runs=True,
        )
        emission_tracker.start()
        self._cc_trackers[key] = emission_tracker

    def _stop_cc_tracker(self, key: str) -> None:
        emission_tracker: EmissionsTracker = self._cc_trackers.pop(key, None)
        if emission_tracker is None:
            return
        emission_tracker.stop()
        emission_data = getattr(emission_tracker, "final_emissions_data", None)
        if emission_data is not None:
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
        self._start_cc_tracker(key)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        """LangChain callback: stop timing and CodeCarbon tracking."""
        key = str(run_id)
        if key in self._starts:
            self.total_time += time.perf_counter() - self._starts.pop(key)
        self._stop_cc_tracker(key)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self.on_llm_end(response=error, run_id=run_id, **kwargs)


__all__ = ["LLMCallAccumulator", "initialize_tracking"]
