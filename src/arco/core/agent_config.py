import dataclasses
import logging
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml

from .exceptions import ConfigException

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Configuration for a single agent execution.

    :ivar provider: Model provider for this specific agent.
    :ivar model: The LLM model used for this agent.
          n Number of best-of-N runs (default 1).
    :ivar bon_parameter: Which sampling parameter to vary (``"temperature"``,
         ``"top_p"``, or ``"top_k"``).
    :ivar temp_min: Minimum sampling temperature.
    :ivar temp_max: Maximum sampling temperature.
    :ivar top_p_min: Top-p sampling lower bound.
    :ivar top_p_max: Top-p sampling upper bound.
    :ivar top_k_min: Top-k sampling lower bound (skipped for OpenAI).
    :ivar top_k_max: Top-k sampling upper bound (skipped for OpenAI).
    :ivar max_tokens: Maximum tokens for LLM generation (default 2000).
    :ivar num_beams: Beam search width (1 = greedy/disabled, skipped for OpenAI).
    :ivar no_repeat_ngram_size: Prevent repeating n-grams (skipped for OpenAI).
    :ivar cot_n: Number of chain-of-thought refinement iterations (default 1).
    :ivar enable_budget_controller: Whether the ARCO budget controller is active.
    """

    # Optional per-step LLM overrides
    _DUMMY_STR = "_DUMMY_STR_"
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
    num_beams: int = (
        1  # Beam search width (1 = greedy/disabled); skipped for OpenAI provider
    )
    no_repeat_ngram_size: int | None = (
        None  # Prevent repeating n-grams of this size; skipped for OpenAI provider
    )

    # CoT iterative refinement
    cot_n: int = 1

    # ARCO parameters
    enable_budget_controller: bool | None = None

    @classmethod
    def from_yaml(
        cls, yaml_path: str, agent_name: str, inherit_globals_from: Config | None = None
    ) -> AgentConfig:
        """Load an agent's configuration from a YAML file.

        Looks for the agent under either the ``agents`` key (run configs)
        or the ``defaults`` key (benchmark configs).  Fields not set in
        the YAML inherit from the global :class:`Config` if provided.

        :param yaml_path: Path to the YAML configuration file.
        :param agent_name: The agent type name (e.g. ``"Retriever"``).
        :param inherit_globals_from: Optional global config to inherit
            provider, model, etc. from.
        :returns: A new :class:`AgentConfig` instance.
        :raises ConfigException: If the YAML file cannot be read.
        """
        logger.debug(f"Loading config for {agent_name} from {yaml_path}")
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        agents_section = raw.get("agents", {})  # for run configs
        if agents_section == {}:  # for benchmark configs
            agents_section = raw.get("defaults", {})

        if agent_name in agents_section:
            agent_dict = dict(agents_section[agent_name])
        else:
            agent_dict = {}

        config = cls.from_dict(agent_dict)

        if inherit_globals_from:
            config = config._inherit_from_config(inherit_globals_from)

        return config._normalize_ranges()

    @classmethod
    def from_config(cls, config: Config) -> AgentConfig:
        """Create an AgentConfig that inherits all fields from a global :class:`Config`.

        :param config: The global config to inherit from.
        :returns: A new :class:`AgentConfig` instance.
        """
        return cls()._inherit_from_config(config=config)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Create an AgentConfig from a dict (for deserialization).

        Unknown keys are silently ignored.

        :param data: Dictionary of field names to values.
        :returns: A new :class:`AgentConfig` instance.
        """
        valid_keys = [f.name for f in dataclasses.fields(AgentConfig)]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def set(self, **kwargs) -> AgentConfig:
        """Return a new AgentConfig with the given fields replaced.

        :param kwargs: Field names and their new values.
        :returns: A new :class:`AgentConfig` instance.
        """
        valid_keys = {f.name for f in dataclasses.fields(AgentConfig)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return replace(self, **filtered)

    def update(self, update_dict: dict[str, Any]) -> AgentConfig:
        """Return a new AgentConfig with fields from *update_dict* applied.

        Unknown keys are silently ignored.  Ranges are normalised after
        the update.

        :param update_dict: Dictionary of field names to values.
        :returns: A new :class:`AgentConfig` instance.
        """
        return self.set(**update_dict)._normalize_ranges()

    def get_candidate_params(self) -> list[tuple[float, float | None, int | None]]:
        """Generate ``(temperature, top_p, top_k)`` tuples for each best-of-N candidate.

        The parameter selected by :attr:`bon_parameter` is varied linearly
        across *n* steps; the other two are fixed at their minimum values.

        :returns: A list of parameter tuples, one per candidate.
        :raises ConfigException: If ``bon_parameter`` is ``top_p`` or ``top_k``
            but the corresponding min/max values are ``None``.
        """
        if self.n <= 1:
            return [(self.temp_min, self.top_p_min, self.top_k_min)]
        if self.bon_parameter == "top_p":
            if self.top_p_min is None or self.top_p_max is None:
                raise ConfigException(
                    "Cannot generate candidates if top_p_min or top_p_max are None"
                )
            top_ps = np.linspace(self.top_p_min, self.top_p_max, self.n).tolist()
            return [(self.temp_min, p, self.top_k_min) for p in top_ps]
        if self.bon_parameter == "top_k":
            if self.top_k_min is None or self.top_k_max is None:
                raise ConfigException(
                    "Cannot generate candidates if top_k_min or top_p_max are None"
                )
            top_ks = [
                int(k) for k in np.linspace(self.top_k_min, self.top_k_max, self.n)
            ]
            return [(self.temp_min, self.top_p_min, k) for k in top_ks]
        # default: temperature
        temps = np.linspace(self.temp_min, self.temp_max, self.n).tolist()
        return [(t, self.top_p_min, self.top_k_min) for t in temps]

    def _inherit_from_config(self, config: Config) -> AgentConfig:
        """Return a new config with unset fields inherited from the global config."""
        kwargs = {}
        if self.provider == self._DUMMY_STR:
            kwargs["provider"] = config.default_provider
        if self.model == self._DUMMY_STR:
            kwargs["model"] = config.default_model
        if self.provider_judge == self._DUMMY_STR:
            kwargs["provider_judge"] = config.default_provider_judge
        if self.model_judge == self._DUMMY_STR:
            kwargs["model_judge"] = config.default_model_judge
        if self.enable_budget_controller is None:
            kwargs["enable_budget_controller"] = config.enable_budget_controller
        if kwargs:
            return replace(self, **kwargs)
        return self

    def _normalize_ranges(self) -> AgentConfig:
        """Return a new config with normalised min/max ranges."""
        kwargs = {}
        if self.n == 1:
            if self.temp_max != self.temp_min:
                kwargs["temp_max"] = self.temp_min
            if self.top_k_max != self.top_k_min:
                kwargs["top_k_max"] = self.top_k_min
            if self.top_p_max != self.top_p_min:
                kwargs["top_p_max"] = self.top_p_min
        else:
            if self.temp_max < self.temp_min:
                kwargs["temp_max"] = self.temp_min
            if self.top_k_min and self.top_k_max and self.top_k_max < self.top_k_min:
                kwargs["top_k_max"] = self.top_k_min
            if self.top_p_min and self.top_p_max and self.top_p_max < self.top_p_min:
                kwargs["top_p_max"] = self.top_p_min
        if kwargs:
            return replace(self, **kwargs)
        return self

    def __rich_repr__(self):
        # Rich automatically detects this method when you pass the object to Pretty()
        for key, value in asdict(self).items():
            if value is not None:
                yield key, value


__all__ = ["AgentConfig"]
