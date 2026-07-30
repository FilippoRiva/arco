import dataclasses
import logging
import random
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

import yaml

from .agent_config import AgentConfig

if TYPE_CHECKING:
    from . import AgentType

logger = logging.getLogger(__name__)


def _generate_readable_id():
    prefixes = [
        "querying",
        "fetching",
        "ingesting",
        "indexing",
        "relational",
        "metered",
        "gauged",
        "streaming",
        "loaded",
        "thrifty",
        "augmented",
        "expanded",
        "parallel",
        "optimized",
        "pruned",
        "branched",
        "ranked",
        "scoring",
        "scored",
        "weighted",
        "top-k",
        "plotting",
        "rendering",
        "mapping",
        "vivid",
        "vectorized",
    ]
    nouns = [
        "schema",
        "pipeline",
        "dataset",
        "ledger",
        "buffer",
        "warehouse",
        "beam",
        "frontier",
        "trajectory",
        "node",
        "nexus",
        "pivot",
        "cascade",
        "canvas",
        "matrix",
        "tensor",
        "figure",
        "chart",
        "graph",
        "palette",
    ]
    number = random.randint(100, 999)
    return f"{random.choice(prefixes)}-{random.choice(nouns)}-{number}"


@dataclass(frozen=True)
class Config:
    """Global and per-agent configuration for a workflow run.

    :ivar workflow: The workflow identifier (e.g. ``"sales"``).
    :ivar prompt: The user prompt for this run.
    :ivar run_id: Unique identifier for this run (auto-generated if not set).
    :ivar enable_budget_controller: Whether the ARCO budget controller is active.
    :ivar default_provider: The default LLM provider (``"openai"``, ``"ollama"``,
          or ``"openrouter"``).
    :ivar default_model: The default model identifier.
    :ivar default_provider_judge: The default provider for LLM-as-a-judge.
    :ivar default_model_judge: The default model for LLM-as-a-judge.
    :ivar ollama_url: Base URL for the Ollama server.
    :ivar enable_storage: Whether to persist state artifacts to disk.
    :ivar save_dir: Directory for output artifacts.
    :ivar enable_codecarbon: Whether to enable CodeCarbon energy tracking.
    :ivar agent_configs: Per-agent configuration dict, keyed by :class:`AgentType`.
    :ivar config_path: Path to the YAML file this config was loaded from.
    """

    workflow: str = ""
    prompt: str = ""
    run_id: str = field(default_factory=lambda: _generate_readable_id())
    enable_budget_controller: bool = True
    default_provider: Literal["openai", "ollama", "openrouter"] = "openai"
    default_model: str = "gpt-4o-mini"
    default_provider_judge: Literal["openai", "ollama", "openrouter"] = "openai"
    default_model_judge: str = "gpt-4o-mini"
    ollama_url: str = "http://localhost:11434"
    enable_storage: bool = False
    save_dir: str = "./output"
    enable_codecarbon: bool = False
    agent_configs: dict[AgentType, AgentConfig] = field(default_factory=dict)
    config_path: str | None = None

    def update_prompt(self, prompt: str) -> Config:
        """Return a new config with the prompt replaced and a fresh run ID.

        :param prompt: The new prompt string.
        :returns: A new :class:`Config` instance.
        """
        temp = self._shuffle_id()
        return dataclasses.replace(temp, prompt=prompt)

    def copy(self) -> Config:
        """Return a deep copy of this configuration.

        :returns: A new :class:`Config` instance with all fields deeply copied.
        """
        from copy import deepcopy

        return deepcopy(self)

    def hydrate_agent_configs(self, agent_list: list[AgentType] | None = None):
        """Populate :attr:`agent_configs` from the registered agent types.

        For each agent type, loads the per-agent config from the YAML file
        (if available) or inherits from the global defaults.

        :param agent_list: List of agent types to hydrate.  If ``None``,
            all registered :class:`AgentType` values are used.
        """
        logger.info(f"Hydrating config for : {[agent.value for agent in agent_list]!s}")
        from .agent_type import AgentType

        self.agent_configs.clear()

        for agent_type in agent_list if agent_list else AgentType.all():
            if self.config_path:
                agent_cfg = AgentConfig.from_yaml(
                    self.config_path, agent_type.value, inherit_globals_from=self
                )
            else:
                agent_cfg = AgentConfig.from_config(self)
            self.agent_configs[agent_type] = agent_cfg

    def set(self, **kwargs) -> Config:
        """Return a new config with the given fields replaced.

        :param kwargs: Field names and their new values.
        :returns: A new :class:`Config` instance.
        """
        return replace(self, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> Config:
        """Load configuration from a YAML file.

        The YAML file should have a ``global`` section with any
        :class:`Config` fields to override.  Per-agent settings are
        loaded separately via :meth:`hydrate_agent_configs`.

        :param yaml_path: Path to the YAML configuration file.
        :returns: A new :class:`Config` instance.
        """
        logger.info(f"Loading configs from {yaml_path}")
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        global_section = raw.get("global", {})

        global_params: dict[str, Any] = {
            "config_path": yaml_path,
        }

        for field_meta in fields(Config):
            if (
                field_meta.name in global_section
                and field_meta.name not in global_params
            ):
                global_params[field_meta.name] = global_section[field_meta.name]

        instance = cls(**global_params)
        return instance

    def generate_benchmark_configs(self, yaml_path: str) -> list[dict[str, Any]]:
        """Generate a list of run configurations from a benchmark YAML file.

        Reads the ``runs`` section of the benchmark config and creates a
        deep copy of this config for each run, applying any per-run
        ``changes``.

        :param yaml_path: Path to a benchmark configuration YAML file.
        :returns: A list of dicts, each with keys ``name``, ``description``,
            ``config``, and ``changes``.
        """
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        runs = raw.get("runs")

        full_benchmark_config_list = []

        for run in runs:
            changes = run.get("changes", [])

            run_config = self.copy()

            for agent in changes:
                from .agent_type import AgentType

                agent_type = AgentType(agent)
                run_config.agent_configs[agent_type].update(changes.get(agent))

            single_run_config_dict = {
                "name": run.get("name"),
                "description": run.get("description"),
                "config": run_config,
                "changes": changes,
            }

            full_benchmark_config_list.append(single_run_config_dict)
        return full_benchmark_config_list

    def _shuffle_id(self) -> Config:
        """Return a new config with a fresh run ID."""
        return replace(self, run_id=_generate_readable_id())


__all__ = ["Config"]
