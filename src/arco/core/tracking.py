import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arco.core import Config

import logging
from collections import defaultdict

from codecarbon import EmissionsTracker
from langchain_core.callbacks import BaseCallbackHandler

logging.getLogger("codecarbon").setLevel(logging.ERROR)  # Hide codecarbon warnings

tracker: EmissionsTracker | None = None


def initialize_tracking(config: Config):
    if not config.enable_codecarbon:
        return

    global tracker
    if tracker is not None:
        return

    codecarbon_dir = os.path.join(config.save_dir or "./output", "codecarbon")
    os.makedirs(codecarbon_dir, exist_ok=True)
    tracker = EmissionsTracker(
        project_name="SalesDataAgent",
        output_dir=codecarbon_dir,
        save_to_file=True,
        measure_power_secs=1,
        log_level="error",
        allow_multiple_runs=True,
        experiment_id=config.run_id
    )
    # LLM Emission Tracking
    LLMCallAccumulator.enable(codecarbon_dir)


def start_tracking():
    global tracker
    if tracker is None:
        return
    tracker.start()


def stop_tracking():
    global tracker
    if tracker is None:
        return
    tracker.stop()


def is_enabled():
    global tracker
    return tracker is not None


def get_energy_dict() -> dict:
    global tracker
    if tracker is None:
        return {}
    ed = tracker.final_emissions_data
    energy_dict = {
        "energy_consumed_kwh": ed.energy_consumed,
        "cpu_energy_kwh": ed.cpu_energy,
        "gpu_energy_kwh": ed.gpu_energy,
        "ram_energy_kwh": ed.ram_energy,
        "emissions_kg_co2": ed.emissions,
        "cpu_power_w": ed.cpu_power,
        "gpu_power_w": ed.gpu_power,
        "duration_sec": ed.duration,
    }
    return energy_dict


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

    def __init__(self, name: str) -> None:
        super().__init__()
        self._starts: dict[str, float | int] = {}
        self._cc_trackers: dict[str, Any] = {}
        self.total_time: float | int = 0.0
        self._cc_output_dir: str | None = os.path.join(LLMCallAccumulator._save_dir,
                                                       name) if LLMCallAccumulator._save_dir else None
        self._enabled: bool = LLMCallAccumulator._enabled
        # Accumulated energy across all invoke() calls for this step
        self.energy_dict: dict[str, float | int] = defaultdict(float)

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
                self.energy_dict["energy_consumed_kwh"] += getattr(_ed, "energy_consumed", 0.0) or 0.0
                self.energy_dict["cpu_energy_kwh"] += getattr(_ed, "cpu_energy", 0.0) or 0.0
                self.energy_dict["gpu_energy_kwh"] += getattr(_ed, "gpu_energy", 0.0) or 0.0
                self.energy_dict["ram_energy_kwh"] += getattr(_ed, "ram_energy", 0.0) or 0.0
                self.energy_dict["emissions_kg_co2"] += getattr(_ed, "emissions", 0.0) or 0.0
        except Exception as _e:
            print(f"[CodeCarbon] per-invoke tracker stop failed: {_e}")

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
