"""Configuration classes for per-step hyperparameter control.

This module provides type-safe, serializable configuration objects for
controlling agent execution at the step level.
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Literal, Dict, Any, List, Tuple
import os
import numpy as np


@dataclass
class StepConfig:
    """Configuration for a single agent step execution.

    Attributes:
        n: Number of best-of-n runs for this step (default 1)
        temp_min: Minimum temperature for sampling (default 0.1)
        temp_max: Maximum temperature for sampling (default 0.1)
        max_tokens: Maximum tokens for LLM generation (default 2000)
        top_p: Top-p sampling parameter (default 1.0)
        top_k: Top-k sampling parameter (default None, skipped for OpenAI)
        num_beams: Beam search width; 1 = greedy/disabled (default 1, skipped for OpenAI)
        no_repeat_ngram_size: Prevent repeating n-grams of this size (default None, skipped for OpenAI)
        eval_fn: Callable that scores a result, signature: (result: Dict, state: State) -> float
        batch_eval_fn: Callable that scores all N results at once, signature: (results: List[Dict], state: Dict) -> List[float]
        selection_fn: Callable that picks best from N scores (default: argmax)
        use_cache: Whether to check cache for this step (default True)
        cache_mode: Cache behavior - "auto", "skip", or "force_fresh"
        enabled: Whether this step runs at all (default True)
        step_name: Name identifier for this step
    """
    # Best-of-n sampling parameters
    n: int = 1
    temp_min: float = 0.1
    temp_max: float = 0.1

    # LLM generation parameters
    max_tokens: int = 2000
    top_p: float = 1.0
    top_k: Optional[int] = None  # Top-k sampling; skipped for OpenAI provider
    num_beams: int = 1  # Beam search width (1 = greedy/disabled); skipped for OpenAI provider
    no_repeat_ngram_size: Optional[int] = None  # Prevent repeating n-grams of this size; skipped for OpenAI provider

    # Evaluation and selection (not serialized)
    eval_fn: Optional[Callable] = None
    batch_eval_fn: Optional[Callable] = None
    selection_fn: Optional[Callable] = None

    # Ground-truth evaluation for tracking/logging only (never used for selection)
    gt_eval_fn: Optional[Callable] = None

    # Caching control
    use_cache: bool = True
    cache_mode: Literal["auto", "skip", "force_fresh"] = "auto"

    # CoT iterative refinement
    cot_n: int = 1

    # Metadata
    enabled: bool = True
    step_name: str = ""

    def __post_init__(self):
        """Set default selection function if not provided."""
        if self.selection_fn is None:
            self.selection_fn = lambda scores: int(np.argmax(scores)) if scores else 0

    def get_temperatures(self) -> List[float]:
        """Generate temperature values for best-of-n sampling."""
        if self.n <= 1:
            return [self.temp_min]
        return np.linspace(self.temp_min, self.temp_max, self.n).tolist()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (excluding non-serializable callables)."""
        d = asdict(self)
        # Remove non-serializable functions
        d.pop('eval_fn', None)
        d.pop('batch_eval_fn', None)
        d.pop('selection_fn', None)
        d.pop('gt_eval_fn', None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepConfig':
        """Create StepConfig from dict (for deserialization)."""
        # Filter out unknown keys and non-serializable fields
        valid_keys = {
            'n', 'temp_min', 'temp_max', 'max_tokens', 'top_p', 'top_k',
            'num_beams', 'no_repeat_ngram_size',
            'use_cache', 'cache_mode', 'enabled', 'step_name', 'cot_n'
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class AgentConfig:
    """Complete agent configuration with per-step settings.

    Holds configurations for all 4 agent steps plus global settings.
    Each step has its own StepConfig that controls best-of-n sampling,
    temperature, evaluation, and caching behavior.

    Attributes:
        decide_tool: Config for the routing/decision step
        lookup_sales_data: Config for SQL generation and data retrieval
        analyzing_data: Config for data analysis generation
        create_visualization: Config for chart configuration and code generation
        model: LLM model name (e.g., "llama3.2:3b")
        provider: LLM provider ("ollama" or "openai")
        ollama_url: Ollama server URL
        openai_api_key: OpenAI API key (if using openai provider)
    """
    # Step-specific configurations
    decide_tool: StepConfig = field(default_factory=lambda: StepConfig(
        step_name="decide_tool",
        n=1,  # Routing doesn't need best-of-n
        use_cache=False  # Routing should always run fresh
    ))

    lookup_sales_data: StepConfig = field(default_factory=lambda: StepConfig(
        step_name="lookup_sales_data",
        n=1,
        temp_min=0.1,
        temp_max=0.3
    ))

    analyzing_data: StepConfig = field(default_factory=lambda: StepConfig(
        step_name="analyzing_data",
        n=1,
        temp_min=0.1,
        temp_max=0.7,
        max_tokens=3000
    ))

    create_visualization: StepConfig = field(default_factory=lambda: StepConfig(
        step_name="create_visualization",
        n=1,
        temp_min=0.1,
        temp_max=0.5
    ))

    # Global LLM settings
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    ollama_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None

    def get_step_config(self, step_name: str) -> StepConfig:
        """Get configuration for a specific step by name.

        Args:
            step_name: Name of the step (e.g., "lookup_sales_data")

        Returns:
            StepConfig for the requested step, or a default StepConfig
            if the step name is not recognized.
        """
        config = getattr(self, step_name, None)
        if isinstance(config, StepConfig):
            return config
        # Return default config for unknown steps
        return StepConfig(step_name=step_name)

    def set_step_config(self, step_name: str, config: StepConfig) -> None:
        """Set configuration for a specific step.

        Args:
            step_name: Name of the step
            config: StepConfig to set
        """
        if hasattr(self, step_name):
            setattr(self, step_name, config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire configuration to dict."""
        return {
            "decide_tool": self.decide_tool.to_dict(),
            "lookup_sales_data": self.lookup_sales_data.to_dict(),
            "analyzing_data": self.analyzing_data.to_dict(),
            "create_visualization": self.create_visualization.to_dict(),
            "model": self.model,
            "provider": self.provider,
            "ollama_url": self.ollama_url,
            # Note: openai_api_key intentionally excluded for security
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from dict (for deserialization)."""
        config = cls()

        # Load step configs
        for step_name in ['decide_tool', 'lookup_sales_data', 'analyzing_data', 'create_visualization']:
            if step_name in data:
                step_config = StepConfig.from_dict(data[step_name])
                setattr(config, step_name, step_config)

        # Load global settings
        if 'model' in data:
            config.model = data['model']
        if 'provider' in data:
            config.provider = data['provider']
        if 'ollama_url' in data:
            config.ollama_url = data['ollama_url']

        return config

    def copy(self) -> 'AgentConfig':
        """Create a deep copy of this configuration."""
        from copy import deepcopy
        return deepcopy(self)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> Tuple['AgentConfig', Dict[str, Any], Any]:
        """Load configuration from a YAML file.

        The YAML file should have sections: agent, steps, schema, run, tracing.
        Schema table definitions are stored in separate per-table YAML files,
        referenced by path in schema.tables.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Tuple of (AgentConfig, run_params dict, DatabaseSchema or None).
            run_params includes keys: prompt, visualization_goal, lookup_only,
            no_vis, run_id, save_dir, save_results, reuse_from, enable_codecarbon,
            and a 'tracing' sub-dict.
        """
        import yaml
        from .schema import DatabaseSchema, TableSchema, ColumnSchema

        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        config_dir = os.path.dirname(os.path.abspath(yaml_path))

        # --- Build AgentConfig ---
        agent_section = raw.get('agent', {})
        steps_section = raw.get('steps', {})

        config = cls()
        config.model = agent_section.get('model', config.model)
        config.provider = agent_section.get('provider', config.provider)
        config.ollama_url = agent_section.get('ollama_url', config.ollama_url)
        config.openai_api_key = agent_section.get(
            'openai_api_key', os.environ.get('OPENAI_API_KEY')
        )

        # Track per-step eval settings before from_dict filters them out
        step_eval_settings: Dict[str, str] = {}
        for step_name in ['decide_tool', 'lookup_sales_data', 'analyzing_data', 'create_visualization']:
            if step_name in steps_section:
                step_data = dict(steps_section[step_name])
                step_eval_settings[step_name] = step_data.pop('eval', 'default')
                step_data.setdefault('step_name', step_name)
                setattr(config, step_name, StepConfig.from_dict(step_data))

        # --- Build DatabaseSchema by discovering *_schema.yaml files ---
        schema = None
        schema_section = raw.get('schema', {})
        data_dir = schema_section.get('data_dir')
        if data_dir:
            import glob as _glob

            if not os.path.isabs(data_dir):
                data_dir = os.path.join(config_dir, data_dir)
            data_dir = os.path.abspath(data_dir)

            schema_files = sorted(_glob.glob(os.path.join(data_dir, '*_schema.yaml')))
            tables = []
            for table_path in schema_files:
                with open(table_path, 'r') as tf:
                    t = yaml.safe_load(tf)

                columns = [
                    ColumnSchema(
                        name=c['name'],
                        description=c.get('description', c['name']),
                        data_type=c.get('data_type', 'VARCHAR'),
                        example_values=c.get('example_values'),
                        nullable=c.get('nullable', True),
                    )
                    for c in t.get('columns', [])
                ]

                # Resolve file_path relative to schema file directory
                schema_dir = os.path.dirname(table_path)
                file_path = t['file_path']
                if not os.path.isabs(file_path):
                    file_path = os.path.join(schema_dir, file_path)

                tables.append(TableSchema(
                    name=t['name'],
                    description=t.get('description', t['name']),
                    file_path=file_path,
                    columns=columns,
                ))

            if tables:
                schema = DatabaseSchema(
                    tables=tables,
                    compact_threshold=schema_section.get('compact_threshold', 5),
                )

        # --- Extract run params ---
        run_section = raw.get('run', {})
        agent_mode = run_section.get('agent_mode', 'full')
        run_params: Dict[str, Any] = {
            'prompt': run_section.get('prompt', ''),
            'visualization_goal': run_section.get('visualization_goal'),
            'lookup_only': agent_mode == 'lookup_only',
            'no_vis': agent_mode in ('lookup_only', 'analysis'),
            'run_id': run_section.get('run_id'),
            'save_dir': run_section.get('save_dir'),
            'enable_codecarbon': run_section.get('enable_codecarbon', False),
            'save_results': run_section.get('save_results', False),
            'reuse_from': run_section.get('reuse_from'),
            'step_overrides': run_section.get('step_overrides'),
            'interactive_config': run_section.get('interactive_config', False),
        }

        # --- Default evaluation functions (attached unless eval: "none") ---
        from .utils import (
            make_csv_evaluator_no_gt,
            make_text_evaluator_no_gt,
            make_vis_evaluator_no_gt,
        )

        if step_eval_settings.get('lookup_sales_data', 'default') != 'none':
            config.lookup_sales_data.batch_eval_fn = make_csv_evaluator_no_gt()

        if step_eval_settings.get('analyzing_data', 'default') != 'none':
            config.analyzing_data.eval_fn = make_text_evaluator_no_gt(
                judge_model=config.model,
                provider=config.provider,
                ollama_url=config.ollama_url,
                openai_api_key=config.openai_api_key,
            )

        if step_eval_settings.get('create_visualization', 'default') != 'none':
            config.create_visualization.eval_fn = make_vis_evaluator_no_gt(
                judge_model=config.model,
                provider=config.provider,
                ollama_url=config.ollama_url,
                openai_api_key=config.openai_api_key,
            )

        # --- Ground truth config (tracking only, never steers selection) ---
        gt_section = raw.get('ground_truth', {})
        if gt_section:
            from .utils import (
                make_csv_evaluator_gt,
                make_csv_evaluator_no_gt,
                make_text_evaluator_gt,
                make_text_evaluator_no_gt,
                make_vis_evaluator_gt,
                make_vis_evaluator_no_gt,
            )

            gt_csv_path = gt_section.get('csv_path')
            if gt_csv_path:
                if not os.path.isabs(gt_csv_path):
                    gt_csv_path = os.path.join(config_dir, gt_csv_path)
                config.lookup_sales_data.gt_eval_fn = make_csv_evaluator_gt(gt_csv_path)
                # Use consensus-based evaluator for actual selection
                config.lookup_sales_data.batch_eval_fn = make_csv_evaluator_no_gt()

            gt_analysis = gt_section.get('analysis_text')
            if gt_analysis:
                config.analyzing_data.gt_eval_fn = make_text_evaluator_gt(
                    ground_truth_text=gt_analysis,
                )
                # Use no-GT judge for actual selection
                config.analyzing_data.eval_fn = make_text_evaluator_no_gt()

            gt_vis_config = gt_section.get('vis_config')
            gt_vis_code = gt_section.get('vis_code')
            if gt_vis_config and gt_vis_code:
                config.create_visualization.gt_eval_fn = make_vis_evaluator_gt(
                    ground_truth_config=gt_vis_config,
                    ground_truth_code=gt_vis_code,
                )
                # Use no-GT judge for actual selection
                config.create_visualization.eval_fn = make_vis_evaluator_no_gt()

        # --- Tracing config (passed separately to SalesDataAgent.__init__) ---
        tracing_section = raw.get('tracing', {})
        run_params['tracing'] = {
            'enabled': tracing_section.get('enabled', False),
            'phoenix_endpoint': tracing_section.get('phoenix_endpoint'),
            'phoenix_api_key': tracing_section.get('phoenix_api_key'),
            'project_name': tracing_section.get('project_name', 'evaluating-agent'),
        }

        return config, run_params, schema
