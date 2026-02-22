"""Configuration classes for per-step hyperparameter control.

This module provides type-safe, serializable configuration objects for
controlling agent execution at the step level.
"""

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Literal, Dict, Any, List
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
        eval_fn: Callable that scores a result, signature: (result: Dict, state: State) -> float
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

    # Evaluation and selection (not serialized)
    eval_fn: Optional[Callable] = None
    selection_fn: Optional[Callable] = None

    # Caching control
    use_cache: bool = True
    cache_mode: Literal["auto", "skip", "force_fresh"] = "auto"

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
        d.pop('selection_fn', None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StepConfig':
        """Create StepConfig from dict (for deserialization)."""
        # Filter out unknown keys and non-serializable fields
        valid_keys = {
            'n', 'temp_min', 'temp_max', 'max_tokens', 'top_p',
            'use_cache', 'cache_mode', 'enabled', 'step_name'
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
