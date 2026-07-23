import dataclasses
from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any, TYPE_CHECKING

import numpy as np
import yaml

from .exceptions import ConfigException

if TYPE_CHECKING:
    from .config import Config


@dataclass
class AgentConfig:
    """Configuration for a single agent execution.

    Attributes:
        provider: model provider for this specific agent
        model: The LLM model from the provider used for this agent
        n: Number of best-of-n runs for this step (default 1)
        temp_min: Minimum temperature for sampling (default 0.1)
        temp_max: Maximum temperature for sampling (default 0.1)
        max_tokens: Maximum tokens for LLM generation (default 2000)
        top_p_min: Top-p sampling parameter, lower bound (default 1.0)
        top_k_min: Top-k sampling parameter, lower bound (default None, skipped for OpenAI)
        num_beams: Beam search width; 1 = greedy/disabled (default 1, skipped for OpenAI)
        no_repeat_ngram_size: Prevent repeating n-grams of this size (default None, skipped for OpenAI)
        schema: DatabaseSchema used by the retriever
    """
    # Optional per-step LLM overrides
    _DUMMY_STR = "_DUMMY_STR"  # used only for typechecking, the actual value is inherited from ArcoConfig and is always a str
    provider: str = _DUMMY_STR
    model: str = _DUMMY_STR
    provider_judge: str = _DUMMY_STR
    model_judge: str = _DUMMY_STR

    # Best-of-n sampling parameters
    n: int = 1
    bon_parameter: Literal["temperature", "top_p", "top_k"] = "temperature"
    _TEMP = 0.1
    temp_min: float = _TEMP
    temp_max: float = _TEMP
    top_p_min: float | None = None
    top_p_max: float | None = None
    top_k_min: int | None = None  # Top-k sampling; skipped for OpenAI provider
    top_k_max: int | None = None

    # LLM generation parameters
    max_tokens: int = 2000
    num_beams: int = 1  # Beam search width (1 = greedy/disabled); skipped for OpenAI provider
    no_repeat_ngram_size: int | None = None  # Prevent repeating n-grams of this size; skipped for OpenAI provider

    # CoT iterative refinement
    cot_n: int = 1

    # ARCO parameters
    enable_budget_controller: bool | None = None

    def get_candidate_params(self) -> list[tuple[float, float | None, int | None]]:
        """Generate (temperature, top_p, top_k) tuples for each best-of-n candidate.

        The parameter selected by bon_param is varied linearly; the others are fixed.
        """
        if self.n <= 1:
            return [(self.temp_min, self.top_p_min, self.top_k_min)]
        if self.bon_parameter == "top_p":
            if self.top_p_min is None or self.top_p_max is None:
                raise ConfigException("Cannot generate candidates if top_p_min or top_p_max are None")
            top_ps = np.linspace(self.top_p_min, self.top_p_max, self.n).tolist()
            return [(self.temp_min, p, self.top_k_min) for p in top_ps]
        if self.bon_parameter == "top_k":
            if self.top_k_min is None or self.top_k_max is None:
                raise ConfigException("Cannot generate candidates if top_p_min or top_p_max are None")
            top_ks = [int(k) for k in np.linspace(self.top_k_min, self.top_k_max, self.n)]
            return [(self.temp_min, self.top_p_min, k) for k in top_ks]
        # default: temperature
        temps = np.linspace(self.temp_min, self.temp_max, self.n).tolist()
        return [(t, self.top_p_min, self.top_k_min) for t in temps]

    def _inherit_from_config(self, global_config: Config):
        if self.provider == self._DUMMY_STR:
            self.provider = global_config.default_provider
        if self.model == self._DUMMY_STR:
            self.model = global_config.default_model
        if self.provider_judge == self._DUMMY_STR:
            self.provider_judge = global_config.default_provider_judge
        if self.model_judge == self._DUMMY_STR:
            self.model_judge = global_config.default_model_judge
        if self.enable_budget_controller is None:
            self.enable_budget_controller = global_config.enable_budget_controller

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        """Create StepConfig from dict (for deserialization)."""
        # Filter out unknown keys and non-serializable fields
        valid_keys = [f.name for f in dataclasses.fields(AgentConfig)]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str, agent_name, inherit_globals_from: Config | None = None) -> AgentConfig:
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        agents_section = raw.get('agents', {})  # for run configs
        if agents_section == {}:  # for benchmark configs
            agents_section = raw.get('defaults', {})

        if agent_name in agents_section.keys():
            agent_dict = dict(agents_section[agent_name])
        else:
            agent_dict = {}

        config = AgentConfig.from_dict(agent_dict)

        if inherit_globals_from:
            config._inherit_from_config(inherit_globals_from)

        config._normalize_ranges()
        return config

    def update(self, update_dict: dict[str, Any]):
        """Update fields on this AgentConfig from a dict, ignoring unspecified/unknown keys."""
        valid_keys = {f.name for f in dataclasses.fields(AgentConfig)}
        for key, value in update_dict.items():
            if key in valid_keys:
                setattr(self, key, value)
        self._normalize_ranges()

    def _normalize_ranges(self):
        if self.n == 1:
            self.temp_max = self.temp_min
            self.top_k_max = self.top_k_min
            self.top_p_max = self.top_p_min
        else:
            if self.temp_max < self.temp_min:
                self.temp_max = self.temp_min
            if self.top_k_min and self.top_k_max and self.top_k_max < self.top_k_min:
                self.top_k_max = self.top_k_min
            if self.top_p_min and self.top_p_max and self.top_p_max < self.top_p_min:
                self.top_p_max = self.top_p_min

    def __rich_repr__(self):
        # Rich automatically detects this method when you pass the object to Pretty()
        for key, value in asdict(self).items():
            if value is not None:
                yield key, value
