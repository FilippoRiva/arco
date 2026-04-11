"""Random search space for AgentConfig hyperparameters.

Loads a search_space.yaml that defines per-step parameter specs and samples
random AgentConfig instances from it.

Usage:
    from evaluation.search_space import SearchSpace

    space = SearchSpace("evaluation/search_space.yaml")
    configs = space.sample(n_configs=50)          # uses yaml seed
    configs = space.sample(n_configs=3, seed=7)   # override seed
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.config import AgentConfig, StepConfig  # noqa: E402

# Fixed max_tokens per step (not part of the search space)
_STEP_MAX_TOKENS: Dict[str, int] = {
    "lookup_sales_data": 2000,
    "analyzing_data": 3000,
    "create_visualization": 2000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_param(spec: Any, rng: np.random.RandomState) -> Any:
    """Sample one value from a parameter spec.

    Spec formats
    ------------
    scalar            → returned as-is (fixed value)
    list              → uniform random choice from the list
    dict {low, high}  → continuous uniform sample in [low, high]
    """
    if isinstance(spec, list):
        return rng.choice(spec)
    if isinstance(spec, dict) and "low" in spec and "high" in spec:
        return float(rng.uniform(spec["low"], spec["high"]))
    return spec


def _round2(x: float) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------

class SearchSpace:
    """Loads a search_space.yaml and samples random AgentConfig instances.

    Parameters
    ----------
    config_path : str
        Path to the search_space.yaml file.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._global: Dict[str, Any] = raw.get("global", {})
        self._steps: Dict[str, Dict[str, Any]] = raw.get("steps", {})

    @property
    def default_seed(self) -> int:
        return int(self._global.get("seed", 42))

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def _sample_step(
        self,
        step_name: str,
        step_spec: Dict[str, Any],
        rng: np.random.RandomState,
    ) -> StepConfig:
        """Sample one StepConfig for a given step."""
        n = int(_sample_param(step_spec.get("n", 1), rng))
        bon_param = str(_sample_param(step_spec.get("bon_param", "temperature"), rng))
        max_tokens = int(step_spec.get("max_tokens", _STEP_MAX_TOKENS.get(step_name, 2000)))

        sc = StepConfig(step_name=step_name, n=n, bon_param=bon_param, max_tokens=max_tokens)
        sc.use_cache = False  # always fresh in experiments

        if bon_param == "temperature":
            t_min = _round2(_sample_param(step_spec.get("temp_min", 0.0), rng))
            t_max = _round2(_sample_param(step_spec.get("temp_max", 0.5), rng))
            sc.temp_min = min(t_min, t_max)
            sc.temp_max = max(t_min, t_max)
            sc.top_p_min = 1.0
            sc.top_p_max = 1.0

        elif bon_param == "top_p":
            # When varying top_p, temperature is fixed at temp_min
            t_fixed = _round2(_sample_param(step_spec.get("temp_min", 0.1), rng))
            sc.temp_min = t_fixed
            sc.temp_max = t_fixed
            p_min = _round2(_sample_param(step_spec.get("top_p_min", 0.7), rng))
            p_max = _round2(_sample_param(step_spec.get("top_p_max", 1.0), rng))
            sc.top_p_min = min(p_min, p_max)
            sc.top_p_max = max(p_min, p_max)

        else:
            raise ValueError(f"Unknown bon_param: {bon_param!r}")

        return sc

    def sample_one(
        self,
        rng: np.random.RandomState,
        base_config: Optional[AgentConfig] = None,
    ) -> AgentConfig:
        """Sample one AgentConfig from the search space.

        Parameters
        ----------
        rng : np.random.RandomState
            Random state (ensures reproducibility when called in sequence).
        base_config : AgentConfig, optional
            If provided, the returned config inherits model/provider from it.

        Returns
        -------
        AgentConfig with sampled step hyperparameters.
        """
        config = base_config.copy() if base_config is not None else AgentConfig()

        for step_name, step_spec in self._steps.items():
            sc = self._sample_step(step_name, step_spec, rng)
            config.set_step_config(step_name, sc)

        return config

    def sample(
        self,
        n_configs: int,
        seed: Optional[int] = None,
        base_config: Optional[AgentConfig] = None,
    ) -> List[AgentConfig]:
        """Sample *n_configs* AgentConfigs from the search space.

        Parameters
        ----------
        n_configs : int
            Number of configs to sample.
        seed : int, optional
            Random seed; defaults to the value in the YAML (global.seed).
        base_config : AgentConfig, optional
            Base config whose model/provider each sample inherits.

        Returns
        -------
        List of AgentConfig instances (length == n_configs).
        """
        rng = np.random.RandomState(seed if seed is not None else self.default_seed)
        return [self.sample_one(rng, base_config=base_config) for _ in range(n_configs)]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def config_to_record(config_id: int, cfg: AgentConfig) -> Dict[str, Any]:
        """Flatten one AgentConfig into a dict suitable for a DataFrame row."""
        row: Dict[str, Any] = {
            "config_id": config_id,
            "model": cfg.model,
            "provider": cfg.provider,
        }
        for step_name in ["lookup_sales_data", "analyzing_data", "create_visualization"]:
            sc = cfg.get_step_config(step_name)
            p = step_name  # full name as column prefix
            row[f"{p}.n"] = sc.n
            row[f"{p}.bon_param"] = sc.bon_param
            row[f"{p}.temp_min"] = sc.temp_min
            row[f"{p}.temp_max"] = sc.temp_max
            row[f"{p}.top_p_min"] = sc.top_p_min
            row[f"{p}.top_p_max"] = sc.top_p_max
            row[f"{p}.max_tokens"] = sc.max_tokens
        return row

    def configs_to_records(self, configs: List[AgentConfig]) -> List[Dict[str, Any]]:
        """Convert a list of AgentConfigs to a list of flat dicts."""
        return [self.config_to_record(i, cfg) for i, cfg in enumerate(configs)]
