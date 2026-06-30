"""Configuration classes for per-step hyperparameter control.

This module provides type-safe, serializable configuration objects for
controlling agent execution at the step level.
"""
import dataclasses
import random
from dataclasses import dataclass, field, fields, asdict
from typing import Literal, Dict, Any, List, TYPE_CHECKING

import numpy as np
import yaml
from pandas import DataFrame

from .exceptions import ConfigException

if TYPE_CHECKING:
    from .state import AgentType
    from arco.data import DatabaseSchema


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
class ArcoConfig:
    """Complete agent configuration with per-agent settings."""
    # #
    # GLOBAL CONFIGURATION
    # #
    # mandatory(the only mandatory parameters)
    prompt: str
    schema: DatabaseSchema
    # optional
    visualization_goal: str = ""  # a specific goal for visualization
    run_id: str = field(
        default_factory=lambda: generate_readable_id())  # the identifier for this run, generated if not provided
    orchestration_enabled: bool = False  # graph mode, if true the orchestrator is enabled
    empower: bool = True # whether if the arco empowerment is active
    enable_budget_controller: bool = True # whether if the arco budget controller is active
    provider: Literal["openai", "ollama"] = "openai"  # global model provider
    model: str = "gpt-4o-mini"  # the model string
    ollama_url: str = "http://localhost:11434"  # the url to the ollama server
    save_state: bool = False  # toggle for state artifact creation
    use_cache: bool = False  # toggle for cache usage
    cache_mode: Literal["read", "r", "write", "w", "read_write", "rw"] = "rw"  # cache usage mode
    save_dir: str | None = None  # save directory for cache or artifacts
    enable_codecarbon: bool = False  # toggle for codecarbon
    enable_tracing: bool = False
    phoenix_endpoint: str | None = None
    phoenix_project_name: str | None = None

    # #
    # AGENTS CONFIGURATION
    # #
    agent_configs: Dict[AgentType, AgentConfig] = field(default_factory=dict)

    # #
    # TRACING CONFIGURATION
    # #
    tracing: dict = field(default_factory=lambda: {"enabled": False})

    # #
    # NOT CONFIGURABLE FROM YAML
    # #
    config_path: str | None = None  # The path to this config YAML file

    def update_prompt(self, prompt: str, visualization_goal: str | None = None):
        return dataclasses.replace(self, prompt=prompt, visualization_goal=visualization_goal)

    def get_agent_config(self, agent_type: AgentType) -> AgentConfig:
        """Get configuration for a specific agent by type"""
        res = self.agent_configs.get(agent_type)
        if not res:
            raise ConfigException(f"The requested agent_config is missing. Requested Agent: {agent_type.value}")
        return res

    def set_agent_config(self, agent_type: AgentType, config: AgentConfig) -> None:
        """Set configuration for a specific step."""
        self.agent_configs[agent_type] = config

    def copy(self) -> 'ArcoConfig':
        """Create a deep copy of this configuration."""
        from copy import deepcopy
        return deepcopy(self)

    def set_gt(self, gt_data: dict[str, Any]):
        for agent_config in self.agent_configs.values():
            agent_config.set_gt(gt_data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> ArcoConfig:
        """Load configuration from a YAML file.

        The YAML file should have sections: agent, steps, schema, run, tracing.
        Schema table definitions are stored in separate per-table YAML files,
        referenced by path in schema.tables.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Tuple of (AgentConfig, run_params dict, DatabaseSchema or None).
            run_params includes keys: prompt,run_id, save_dir, save_results,
            reuse_from, enable_codecarbon, and a 'tracing' sub-dict.
        """
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        # Fallback to a default instance fields for missing global values
        g = raw.get('global', {})

        from arco.data import DatabaseSchema
        # Set particular configs from the yaml
        global_params = {
            "config_path": yaml_path,
            "schema": DatabaseSchema.from_yaml(yaml_path),
        }

        # Load all the global configs that are overridden in the YAML file
        for field_meta in fields(ArcoConfig):
            if field_meta.name in g:
                if field_meta.name not in global_params.keys():  # avoids redefining schema or other specific params
                    global_params[field_meta.name] = g[field_meta.name]

        if 'prompt' not in global_params.keys():
            raise ConfigException("Prompt should be specified in the yaml")

        # Create intermediate config containing resolved globals so agents can inherit from it
        temp_config = cls(**global_params)  # pyrefly: ignore [missing-argument]

        from arco.core.state import AgentType
        agent_configs = {}
        for agent_type in AgentType:
            agent_cfg = AgentConfig.from_yaml(yaml_path, agent_type.value, inherit_globals_from=temp_config)
            agent_configs[agent_type] = agent_cfg

        # Generate final config
        # pyrefly: ignore [missing-argument]
        return cls(
            **{
                **global_params,
                "agent_configs": agent_configs
            }
        )


@dataclass
class AgentConfig:
    """Configuration for a single agent execution.

    Attributes:
        agent_name: Name identifier for this step
        provider: model provider for this specific agent
        model: The LLM model from the provider used for this agent
        ollama_url: The ollama url for llm instantiation
        n: Number of best-of-n runs for this step (default 1)
        temp_min: Minimum temperature for sampling (default 0.1)
        temp_max: Maximum temperature for sampling (default 0.1)
        max_tokens: Maximum tokens for LLM generation (default 2000)
        top_p_min: Top-p sampling parameter, lower bound (default 1.0)
        top_k_min: Top-k sampling parameter, lower bound (default None, skipped for OpenAI)
        num_beams: Beam search width; 1 = greedy/disabled (default 1, skipped for OpenAI)
        no_repeat_ngram_size: Prevent repeating n-grams of this size (default None, skipped for OpenAI)
        use_cache: Whether to check cache for this step (default True)
        cache_mode: Cache behavior - "auto", "skip", or "force_fresh"
        schema: DatabaseSchema used by the retriever
    """
    agent_name: str

    # Optional per-step LLM overrides
    _DUMMY_STR = "_DUMMY_STR"  # used only for typechecking, the actual value is inherited from ArcoConfig and is always a str
    provider: str = _DUMMY_STR
    model: str = _DUMMY_STR
    ollama_url: str = _DUMMY_STR

    # Best-of-n sampling parameters
    n: int = 1
    bon_parameter: Literal["temperature", "top_p", "top_k"] = "temperature"
    _TEMP = 0.1
    temp_min: float = _TEMP
    temp_max: float = _TEMP

    # LLM generation parameters
    max_tokens: int = 2000
    top_p_min: float | None = None
    top_p_max: float | None = None
    top_k_min: int | None = None  # Top-k sampling; skipped for OpenAI provider
    top_k_max: int | None = None
    num_beams: int = 1  # Beam search width (1 = greedy/disabled); skipped for OpenAI provider
    no_repeat_ngram_size: int | None = None  # Prevent repeating n-grams of this size; skipped for OpenAI provider

    # GT Evaluation configuration
    run_gt_eval: bool = False
    gt_data: DataFrame | None = None  # Retriever gt dataframe
    gt_columns: List[str] | None = None  # Retriever gt_columns for alignment (from gt_data)
    gt_metric = None  # Analyzer evaluation technique
    gt_analysis = None  # Analyzer gt text
    gt_chart_config = None  # Visualizer chart configuration gt
    gt_code = None  # Visualizer code gt
    gt_visual_requirements = None  # Visualizer visual requirements gt

    # Caching control
    use_cache: bool | None = None

    # CoT iterative refinement
    cot_n: int = 1

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

    def _inherit_from_arco_config(self, global_config: ArcoConfig):
        if self.provider == self._DUMMY_STR:
            self.provider = global_config.provider
        if self.model == self._DUMMY_STR:
            self.model = global_config.model
        if self.ollama_url == self._DUMMY_STR:
            self.ollama_url = global_config.ollama_url
        if self.use_cache is None:
            self.use_cache = global_config.use_cache

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        """Create StepConfig from dict (for deserialization)."""
        # Filter out unknown keys and non-serializable fields
        valid_keys = [f.name for f in dataclasses.fields(AgentConfig)]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str, agent_name, inherit_globals_from: ArcoConfig | None = None) -> AgentConfig:
        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        agents_section = raw.get('agents', {})
        if agent_name in agents_section.keys():
            agent_dict = dict(agents_section[agent_name])
            agent_dict.setdefault('agent_name', agent_name)
        else:
            agent_dict = {'agent_name': agent_name}

        config = AgentConfig.from_dict(agent_dict)

        if inherit_globals_from:
            config._inherit_from_arco_config(inherit_globals_from)

        if config.n == 1:
            config.temp_max = config.temp_min
            config.top_k_max = config.top_k_min
            config.top_p_max = config.top_p_min
        else:
            if config.temp_max < config.temp_min:
                config.temp_max = config.temp_min
            if config.top_k_min and config.top_k_max and config.top_k_max < config.top_k_min:
                config.top_k_max = config.top_k_min
            if config.top_p_min and config.top_p_max and config.top_p_max < config.top_p_min:
                config.top_p_max = config.top_p_min

        return config

    def set_gt(self, gt_dict: dict[str, Any]):
        from arco.core.state import AgentType
        if self.agent_name == AgentType.RETRIEVER:
            if "gt_data" in gt_dict.keys():
                import pandas as pd
                csv = pd.io.common.StringIO(gt_dict["gt_data"])
            elif "gt_csv_path" in gt_dict.keys():
                import pandas as pd
                with open(gt_dict["gt_csv_path"], "r", encoding="utf-8") as f:
                    csv = pd.io.common.StringIO(f.read())
            else:
                self.run_gt_eval = False
                return
            self.gt_data = pd.read_csv(csv)
            self.gt_columns = [c.lower() for c in self.gt_data.columns]
            self.run_gt_eval = True
        elif self.agent_name == AgentType.ANALYZER:
            if "gt_analysis" in gt_dict.keys():
                self.gt_analysis = gt_dict["gt_analysis"]
                self.run_gt_eval = True
            if "gt_metric" in gt_dict.keys():
                self.gt_metric = gt_dict["gt_metric"]
        elif self.agent_name == AgentType.VISUALIZER:
            if "gt_chart_config" in gt_dict.keys() and "gt_chart_code" in gt_dict.keys():
                self.gt_chart_config = gt_dict["gt_chart_config"]
                self.gt_code = gt_dict["gt_chart_code"]
                self.run_gt_eval = True
            if "gt_visual_requirements" in gt_dict.keys():
                self.gt_visual_requirements = gt_dict["gt_visual_requirements"]

    def __rich_repr__(self):
        # Rich automatically detects this method when you pass the object to Pretty()
        for key, value in asdict(self).items():
            if value is not None:
                yield key, value
