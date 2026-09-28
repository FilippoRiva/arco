"""Public core API, resolved lazily to keep lightweight imports lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "LLM",
    "Agent",
    "AgentConfig",
    "AgentException",
    "AgentType",
    "Answer",
    "Config",
    "ConfigException",
    "Evaluation",
    "Evaluator",
    "EvaluatorException",
    "ExperimentCatalog",
    "ExperimentSpec",
    "Graph",
    "LLMAnswer",
    "LLMCallAccumulator",
    "ProfilingData",
    "State",
    "StateException",
    "Workflow",
    "WorkflowFactory",
    "check_model_availability",
    "evaluate_state_with_benchmark_entry",
    "get_llm",
    "get_llm_from_config",
    "initialize_tracking",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "LLM": ("llm_tools", "LLM"),
    "Agent": ("agent", "Agent"),
    "AgentConfig": ("agent_config", "AgentConfig"),
    "AgentException": ("exceptions", "AgentException"),
    "AgentType": ("agent_type", "AgentType"),
    "Answer": ("answer", "Answer"),
    "Config": ("config", "Config"),
    "ConfigException": ("exceptions", "ConfigException"),
    "Evaluation": ("evaluator", "Evaluation"),
    "Evaluator": ("evaluator", "Evaluator"),
    "EvaluatorException": ("exceptions", "EvaluatorException"),
    "ExperimentCatalog": ("experiment", "ExperimentCatalog"),
    "ExperimentSpec": ("experiment", "ExperimentSpec"),
    "Graph": ("graph", "Graph"),
    "LLMAnswer": ("llm_tools", "LLMAnswer"),
    "LLMCallAccumulator": ("tracking", "LLMCallAccumulator"),
    "ProfilingData": ("profiling_data", "ProfilingData"),
    "State": ("state", "State"),
    "StateException": ("exceptions", "StateException"),
    "Workflow": ("workflow", "Workflow"),
    "WorkflowFactory": ("workflow", "WorkflowFactory"),
    "check_model_availability": ("llm_tools", "check_model_availability"),
    "evaluate_state_with_benchmark_entry": (
        "evaluator",
        "evaluate_state_with_benchmark_entry",
    ),
    "get_llm": ("llm_tools", "get_llm"),
    "get_llm_from_config": ("llm_tools", "get_llm_from_config"),
    "initialize_tracking": ("tracking", "initialize_tracking"),
}

_SUBMODULES = frozenset(
    {
        "agent",
        "agent_config",
        "agent_type",
        "answer",
        "config",
        "evaluator",
        "exceptions",
        "experiment",
        "graph",
        "llm_tools",
        "profiling_data",
        "state",
        "tracking",
        "workflow",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve a public symbol or core submodule on first access."""
    if name in _EXPORTS:
        module_name, attribute_name = _EXPORTS[name]
        module = import_module(f".{module_name}", package=__name__)
        value = getattr(module, attribute_name)
    elif name in _SUBMODULES:
        value = import_module(f".{name}", package=__name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)


if TYPE_CHECKING:
    from .agent import Agent
    from .agent_config import AgentConfig
    from .agent_type import AgentType
    from .answer import Answer
    from .config import Config
    from .evaluator import Evaluation, Evaluator, evaluate_state_with_benchmark_entry
    from .exceptions import (
        AgentException,
        ConfigException,
        EvaluatorException,
        StateException,
    )
    from .experiment import ExperimentCatalog, ExperimentSpec
    from .graph import Graph
    from .llm_tools import (
        LLM,
        LLMAnswer,
        check_model_availability,
        get_llm,
        get_llm_from_config,
    )
    from .profiling_data import ProfilingData
    from .state import State
    from .tracking import LLMCallAccumulator, initialize_tracking
    from .workflow import Workflow, WorkflowFactory
