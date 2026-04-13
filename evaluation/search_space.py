"""Random search space for AgentConfig hyperparameters.

Loads a search_space.yaml that defines per-step parameter specs and samples
random AgentConfig instances from it.

Provider awareness
------------------
Parameters that are OpenAI-incompatible (top_k, num_beams, no_repeat_ngram_size)
are sampled only when provider != "openai".  The agent's _create_llm already
ignores them for OpenAI, but we avoid polluting the config records with
meaningless values when running against the OpenAI API.

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

# Default max_tokens per step when not specified in YAML
_STEP_MAX_TOKENS: Dict[str, int] = {
    "lookup_sales_data": 2000,
    "analyzing_data": 3000,
    "create_visualization": 2000,
}

# Providers that do NOT support top_k / num_beams / no_repeat_ngram_size
_OPENAI_PROVIDERS = {"openai"}


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
        provider: str = "openai",
    ) -> StepConfig:
        """Sample one StepConfig for a given step.

        Parameters
        ----------
        step_name : str
        step_spec : dict  — parameter specs from YAML for this step.
        rng : RandomState
        provider : str    — LLM provider; controls which parameters are sampled.
            "openai" → top_k / num_beams / no_repeat_ngram_size are NOT sampled
                       (they would be silently ignored by _create_llm anyway).
            anything else → all parameters including non-OpenAI ones are sampled.
        """
        is_openai = provider in _OPENAI_PROVIDERS

        n = int(_sample_param(step_spec.get("n", 1), rng))
        max_tokens = int(_sample_param(
            step_spec.get("max_tokens", _STEP_MAX_TOKENS.get(step_name, 2000)), rng
        ))

        # bon_param choices: base list + non-OpenAI extras when applicable
        bon_choices = list(step_spec.get("bon_param", ["temperature"]))
        if not is_openai:
            bon_choices = bon_choices + list(step_spec.get("bon_param_extra", []))
        bon_param = str(rng.choice(bon_choices))

        sc = StepConfig(step_name=step_name, n=n, bon_param=bon_param, max_tokens=max_tokens)
        sc.use_cache = False  # always fresh in experiments

        # ---- BoN axis and its counterpart (fixed single value) ----
        if bon_param == "temperature":
            # Temperature varies across BoN candidates
            t_min = _round2(_sample_param(step_spec.get("temp_min", 0.0), rng))
            t_max = _round2(_sample_param(step_spec.get("temp_max", 0.5), rng))
            sc.temp_min = min(t_min, t_max)
            sc.temp_max = max(t_min, t_max)
            # top_p fixed for all candidates (reuses top_p_min spec)
            top_p_fixed = _round2(_sample_param(step_spec.get("top_p_min", 1.0), rng))
            sc.top_p_min = top_p_fixed
            sc.top_p_max = top_p_fixed
            # top_k fixed (non-OpenAI only)
            if not is_openai:
                top_k_val = _sample_param(step_spec.get("top_k_min", None), rng)
                sc.top_k_min = int(top_k_val) if top_k_val is not None else None
                sc.top_k_max = sc.top_k_min

        elif bon_param == "top_p":
            # top_p varies across BoN candidates
            p_min = _round2(_sample_param(step_spec.get("top_p_min", 0.7), rng))
            p_max = _round2(_sample_param(step_spec.get("top_p_max", 1.0), rng))
            sc.top_p_min = min(p_min, p_max)
            sc.top_p_max = max(p_min, p_max)
            # temperature fixed for all candidates
            t_fixed = _round2(_sample_param(step_spec.get("temp_min", 0.1), rng))
            sc.temp_min = t_fixed
            sc.temp_max = t_fixed
            # top_k fixed (non-OpenAI only)
            if not is_openai:
                top_k_val = _sample_param(step_spec.get("top_k_min", None), rng)
                sc.top_k_min = int(top_k_val) if top_k_val is not None else None
                sc.top_k_max = sc.top_k_min

        elif bon_param == "top_k":
            # top_k varies across BoN candidates (non-OpenAI only)
            k_min_val = _sample_param(step_spec.get("top_k_min", 20), rng)
            k_max_val = _sample_param(step_spec.get("top_k_max", 100), rng)
            if k_min_val is not None and k_max_val is not None:
                sc.top_k_min = int(min(k_min_val, k_max_val))
                sc.top_k_max = int(max(k_min_val, k_max_val))
            # temperature and top_p fixed for all candidates
            t_fixed = _round2(_sample_param(step_spec.get("temp_min", 0.1), rng))
            sc.temp_min = t_fixed
            sc.temp_max = t_fixed
            top_p_fixed = _round2(_sample_param(step_spec.get("top_p_min", 1.0), rng))
            sc.top_p_min = top_p_fixed
            sc.top_p_max = top_p_fixed

        else:
            raise ValueError(f"Unknown bon_param: {bon_param!r}")

        # ---- cot_n (all providers) ----
        sc.cot_n = int(_sample_param(step_spec.get("cot_n", 1), rng))

        # ---- Non-OpenAI parameters ----
        if not is_openai:
            sc.num_beams = int(_sample_param(step_spec.get("num_beams", 1), rng))
            ngram = _sample_param(step_spec.get("no_repeat_ngram_size", None), rng)
            sc.no_repeat_ngram_size = int(ngram) if ngram is not None else None

        return sc

    def sample_one(
        self,
        rng: np.random.RandomState,
        base_config: Optional[AgentConfig] = None,
        vary_step: Optional[str] = None,
    ) -> AgentConfig:
        """Sample one AgentConfig from the search space.

        Parameters
        ----------
        rng : np.random.RandomState
        base_config : AgentConfig, optional
            Inherited model/provider.  Provider determines which parameters
            are sampled (OpenAI vs non-OpenAI).
        vary_step : str, optional
            If provided, only this step's hyperparameters are sampled.
            Other steps use default StepConfig (n=1).

        Returns
        -------
        AgentConfig with sampled step hyperparameters.
        """
        config = base_config.copy() if base_config is not None else AgentConfig()
        provider = config.provider

        for step_name, step_spec in self._steps.items():
            if vary_step is not None and step_name != vary_step:
                sc = StepConfig(step_name=step_name)
                sc.use_cache = False
            else:
                sc = self._sample_step(step_name, step_spec, rng, provider=provider)
            config.set_step_config(step_name, sc)

        return config

    def sample(
        self,
        n_configs: int,
        seed: Optional[int] = None,
        base_config: Optional[AgentConfig] = None,
        vary_step: Optional[str] = None,
    ) -> List[AgentConfig]:
        """Sample *n_configs* AgentConfigs from the search space."""
        rng = np.random.RandomState(seed if seed is not None else self.default_seed)
        return [self.sample_one(rng, base_config=base_config, vary_step=vary_step) for _ in range(n_configs)]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def config_to_record(config_id: int, cfg: AgentConfig) -> Dict[str, Any]:
        """Flatten one AgentConfig into a dict suitable for a DataFrame row."""
        is_openai = cfg.provider in _OPENAI_PROVIDERS
        row: Dict[str, Any] = {
            "config_id": config_id,
            "model": cfg.model,
            "provider": cfg.provider,
        }
        for step_name in ["lookup_sales_data", "analyzing_data", "create_visualization"]:
            sc = cfg.get_step_config(step_name)
            p = step_name
            row[f"{p}.n"] = sc.n
            row[f"{p}.bon_param"] = sc.bon_param
            row[f"{p}.cot_n"] = sc.cot_n
            row[f"{p}.temp_min"] = sc.temp_min
            row[f"{p}.temp_max"] = sc.temp_max
            row[f"{p}.top_p_min"] = sc.top_p_min
            row[f"{p}.top_p_max"] = sc.top_p_max
            row[f"{p}.max_tokens"] = sc.max_tokens
            # Non-OpenAI parameters: always present in record (None for OpenAI)
            row[f"{p}.top_k_min"] = sc.top_k_min
            row[f"{p}.top_k_max"] = sc.top_k_max
            row[f"{p}.num_beams"] = sc.num_beams
            row[f"{p}.no_repeat_ngram_size"] = sc.no_repeat_ngram_size
        return row

    def configs_to_records(self, configs: List[AgentConfig]) -> List[Dict[str, Any]]:
        """Convert a list of AgentConfigs to a list of flat dicts."""
        return [self.config_to_record(i, cfg) for i, cfg in enumerate(configs)]
