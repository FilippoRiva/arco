"""Configuration classes for per-step hyperparameter control.

This module provides type-safe, serializable configuration objects for
controlling agent execution at the step level.
"""
import dataclasses
import random
from dataclasses import dataclass, field, fields, replace
from typing import Literal, Dict, Any, TYPE_CHECKING

import yaml

from .agent_config import AgentConfig
from .exceptions import ConfigException

if TYPE_CHECKING:
    from . import AgentType


def generate_readable_id():
    prefixes = [
        "querying", "fetching", "ingesting", "indexing", "relational",
        "metered", "gauged", "streaming", "loaded", "thrifty",
        "augmented", "expanded", "parallel", "optimized", "pruned",
        "branched", "ranked", "scoring", "scored", "weighted", "top-k",
        "plotting", "rendering", "mapping", "vivid", "vectorized"
    ]
    nouns = [
        "schema", "pipeline", "dataset", "ledger", "buffer", "warehouse",
        "beam", "frontier", "trajectory", "node", "nexus", "pivot", "cascade",
        "canvas", "matrix", "tensor", "figure", "chart", "graph", "palette"
    ]
    number = random.randint(100, 999)
    return f"{random.choice(prefixes)}-{random.choice(nouns)}-{number}"


@dataclass(frozen=True)
class Config:
    """Complete agent configuration with per-agent settings."""
    # #
    # GLOBAL CONFIGURATION
    # #
    # mandatory
    workflow: str

    # optional
    prompt: str = ""
    run_id: str = field(
        default_factory=lambda: generate_readable_id())  # the identifier for this run, generated if not provided
    enable_budget_controller: bool = True  # whether if the budget controller is active
    default_provider: Literal["openai", "ollama", "openrouter"] = "openai"  # global model provider
    default_model: str = "gpt-4o-mini"  # the model string
    default_provider_judge: Literal["openai", "ollama", "openrouter"] = "openai"
    default_model_judge: str = "gpt-4o-mini"
    ollama_url: str = "http://localhost:11434"  # the url to the ollama server
    enable_storage: bool = False  # toggle for state artifact creation
    save_dir: str = "./output"  # save directory for artifacts
    enable_codecarbon: bool = False  # toggle for codecarbon

    # #
    # AGENTS CONFIGURATION
    # #
    agent_configs: Dict[AgentType, AgentConfig] = field(default_factory=dict)

    # #
    # NOT CONFIGURABLE FROM YAML
    # #
    config_path: str | None = None  # The path to this config YAML file

    def update_prompt(self, prompt: str):
        temp = self._shuffle_id()
        return dataclasses.replace(temp, prompt=prompt)

    def get_agent_config(self, agent_type: AgentType) -> AgentConfig:
        """Get configuration for a specific agent by type"""
        res = self.agent_configs.get(agent_type)
        if not res:
            raise ConfigException(f"The requested agent_config is missing. Requested Agent: {agent_type.value}")
        return res

    def set_agent_config(self, agent_type: AgentType, config: AgentConfig) -> None:
        """Set configuration for a specific step."""
        self.agent_configs[agent_type] = config

    def copy(self) -> 'Config':
        """Create a deep copy of this configuration."""
        from copy import deepcopy
        return deepcopy(self)

    def set_gt(self, gt_data: dict[str, Any]):
        for (agent_type, agent_config) in self.agent_configs.items():
            agent_config.set_gt(gt_data, agent_type)

    def hydrate_agent_configs(self, yaml_path: str):
        """Populate the agent_configs when the agent types are known"""
        from arco.core.state import AgentType
        for agent_type in AgentType.all():
            agent_cfg = AgentConfig.from_yaml(yaml_path, agent_type.value, inherit_globals_from=self)
            self.agent_configs[agent_type] = agent_cfg

    @classmethod
    def from_yaml(cls, yaml_path: str) -> Config:
        """Load configuration from a YAML file.

        The YAML file should have sections: agent, steps, schema, run
        Schema table definitions are stored in separate per-table YAML files,
        referenced by path in schema.tables.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Tuple of (AgentConfig, run_params dict, DatabaseSchema or None).
            run_params includes keys: prompt,run_id, save_dir, save_results,
            reuse_from and enable_codecarbon.
        """
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        global_section = raw.get('global', {})

        # Set particular configs from the yaml (that needs to be instantiated or cannot be derived from the file)
        global_params = {
            "config_path": yaml_path,
        }

        # Load all the global configs that are overridden in the YAML file
        for field_meta in fields(Config):
            if field_meta.name in global_section:
                if field_meta.name not in global_params.keys():  # avoids redefining schema or other specific params
                    global_params[field_meta.name] = global_section[field_meta.name]

        return cls(**global_params)

    def generate_benchmark_configs(self, yaml_path: str) -> list[dict[str, Any]]:
        """Given a Benchmark yaml configuration file (as specified in its schema.json) acts as a factory of
        configurations, used by the benchmark script"""
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        runs = raw.get("runs")

        full_benchmark_config_list = []

        for i, run in enumerate(runs):
            changes = run.get("changes", [])

            run_config = self.copy()

            # set the changes for this specific run configuration
            for agent in changes:
                from arco.core import AgentType
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
        return replace(self, run_id=generate_readable_id())
