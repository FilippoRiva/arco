"""Parameter provider abstraction for runtime step configuration.

This module defines a Protocol for providing step-level configuration at
runtime, enabling dynamic parameter overrides between agent steps. The
abstraction supports:
- Default (static) configuration from YAML
- Interactive terminal-based overrides
- Future LLM-based orchestration
"""

import sys
from copy import deepcopy
from typing import Any, Dict, Protocol

from .config import StepConfig


class ParameterProvider(Protocol):
    """Protocol for providing step configuration at runtime.

    Implementations receive the step name, the YAML-loaded default config,
    and the current agent state, and return a (possibly modified) StepConfig
    to use for that execution.
    """

    def get_step_config(
        self,
        step_name: str,
        default_config: StepConfig,
        state: Dict[str, Any],
    ) -> StepConfig:
        """Return a StepConfig for the given step.

        Args:
            step_name: Which step is about to run (e.g. "lookup_sales_data").
            default_config: The YAML-loaded default config for this step.
            state: Current LangGraph state dict (prompt, prior results, etc.).

        Returns:
            A StepConfig to use for this execution.
        """
        ...


class DefaultProvider:
    """Returns the default config unchanged. Preserves current behavior."""

    def get_step_config(
        self,
        step_name: str,
        default_config: StepConfig,
        state: Dict[str, Any],
    ) -> StepConfig:
        return default_config


# Tunable parameters shown to the user, with their expected types.
_TUNABLE_PARAMS = [
    ("n", int),
    ("temp_min", float),
    ("temp_max", float),
    ("max_tokens", int),
    ("top_p", float),
    ("top_k", int),
    ("cot_n", int),
]

# Steps that should not be overridden interactively.
_SKIP_STEPS = {"decide_tool"}


class TerminalProvider:
    """Interactive terminal provider that prompts for parameter overrides.

    After the decide_tool selects the next step, displays the default
    parameters and asks the user to accept or override them one by one.

    Falls back to DefaultProvider behavior when stdin is not a TTY.
    """

    def __init__(self) -> None:
        self._interactive = sys.stdin.isatty()
        if not self._interactive:
            print("[TerminalProvider] Non-interactive terminal detected, using defaults.")

    def get_step_config(
        self,
        step_name: str,
        default_config: StepConfig,
        state: Dict[str, Any],
    ) -> StepConfig:
        if not self._interactive or step_name in _SKIP_STEPS:
            return default_config

        # Display current defaults
        print(f"\n── [{step_name}] proposed config ──")
        for param_name, _ in _TUNABLE_PARAMS:
            value = getattr(default_config, param_name, None)
            print(f"  {param_name:20s} {value}")
        print()

        # Ask accept/reject
        choice = input("Accept defaults? [Y/n]: ").strip().lower()
        if choice not in ("n", "no"):
            return default_config

        # Prompt each parameter one by one
        new_config = deepcopy(default_config)
        for param_name, param_type in _TUNABLE_PARAMS:
            current_value = getattr(default_config, param_name, None)
            new_value = self._prompt_param(param_name, current_value, param_type)
            setattr(new_config, param_name, new_value)

        # Preserve callable fields from the original config
        new_config.eval_fn = default_config.eval_fn
        new_config.batch_eval_fn = default_config.batch_eval_fn
        new_config.selection_fn = default_config.selection_fn
        new_config.gt_eval_fn = default_config.gt_eval_fn

        return new_config

    @staticmethod
    def _prompt_param(name: str, current_value: Any, param_type: type) -> Any:
        """Prompt for a single parameter, returning current_value on empty input."""
        while True:
            raw = input(f"  {name} [{current_value}]: ").strip()
            if not raw:
                return current_value
            try:
                if param_type is int:
                    if raw.lower() in ("none", "null"):
                        return None
                    return int(raw)
                elif param_type is float:
                    return float(raw)
                else:
                    return param_type(raw)
            except (ValueError, TypeError):
                print(f"    Invalid value for {name} (expected {param_type.__name__}). Try again.")
