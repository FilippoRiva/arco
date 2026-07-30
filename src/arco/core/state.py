import dataclasses
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .agent_type import AgentType
from .answer import Answer
from .config import AgentConfig
from .profiling_data import ProfilingData

logger = logging.getLogger(__name__)


# Immutable dataclass representing the state
@dataclass(frozen=True, slots=True)
class State:
    """Immutable workflow state propagated through the LangGraph.

    :ivar prompt: The original user prompt.
    :ivar run_id: Unique identifier for this run.
    :ivar _agent_configs: Per-agent configuration dict.
    :ivar answers: Ordered list of agent answers produced so far.
    :ivar global_profiling_data: Cumulative profiling data across all steps.
    :ivar agents_profiling_data: Per-agent cumulative profiling data.
    """

    # Original prompt
    prompt: str

    # Run unique identifier
    run_id: str

    # Dynamic Configuration for agents
    agent_configs: Mapping[AgentType, AgentConfig]

    # List of agent's answers
    answers: tuple[Answer] = field(default_factory=tuple)

    # List of metrics profiling the current state
    global_profiling_data: ProfilingData = field(default_factory=ProfilingData)
    agents_profiling_data: Mapping[AgentType, ProfilingData] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def add_answer(self, answer: Answer) -> State:
        """Return a new state with *answer* appended to the answers list.

        Args:
            answer: The answer to append.

        Returns:
            A new state with the answer added.
        """
        return dataclasses.replace(self, answers=(*self.answers, answer))

    def get_last_answer(self, agent_type: AgentType | None = None) -> Answer | None:
        """Retrieve the most recent answer for a specific agent type.

        Args:
            agent_type: The agent type to filter by (e.g. ``"Visualizer"``).
                If ``None``, returns the last answer overall.

        Returns:
            The last matching answer, or ``None``.
        """
        answers = self.answers
        if agent_type:
            return next(
                (item for item in reversed(answers) if item.agent_id == agent_type),
                None,
            )
        return answers[-1] if len(answers) > 0 else None

    def replace_last_answer(self, answer: Answer) -> State:
        """Return a new state with the last answer replaced."""
        last_answer = self.get_last_answer()
        if not last_answer:
            return dataclasses.replace(self, answers=[answer])
        new_answers = tuple([answer for answer in self.answers[:-1]] + [answer])
        return dataclasses.replace(self, answers=new_answers)

    def get_agent_config(self, agent_type: AgentType | None) -> AgentConfig:
        """Return the agent config for any agent type. If the agent is not specified or defined in the list of
        available configs, the default config is returned"""
        if agent_type is not None and agent_type in self.agent_configs:
            return self.agent_configs[agent_type]
        return self.agent_configs["__default__"]

    def get_agents_used(self) -> list[str]:
        """Return the list of agent type names used so far, excluding orchestrator."""
        return [
            answer.agent_id.lower()
            for answer in self.answers
            if answer.agent_id != "Orchestrator"
        ]

    def set_profiling_data(
        self, profiling_data: ProfilingData, agent_type: AgentType
    ) -> State:
        """Attach profiling data and return a new state.

        Accumulates into ``global_profiling_data``, per-agent
        ``agents_profiling_data``, and the last answer's profiling data.
        """
        global_profiling_data = self.global_profiling_data + profiling_data

        agents_profiling_data = {
            k: value.copy() for (k, value) in self.agents_profiling_data.items()
        }
        if agent_type in self.agents_profiling_data:
            agents_profiling_data[agent_type] = (
                agents_profiling_data[agent_type] + profiling_data
            )
        else:
            agents_profiling_data[agent_type] = profiling_data

        new_state = self.replace_last_answer(
            self.get_last_answer(agent_type).set(profiling_data=profiling_data)
        )

        return replace(
            new_state,
            global_profiling_data=global_profiling_data,
            agents_profiling_data=MappingProxyType(agents_profiling_data),
        )

    def to_dict(self) -> dict:
        """Serialize the state to a JSON-compatible dict."""
        return {
            "prompt": self.prompt,
            "run_id": self.run_id,
            "agent_configs": {k: asdict(v) for k, v in self.agent_configs.items()},
            "answers": [a.to_dict() for a in self.answers],
            "global_profiling_data": asdict(self.global_profiling_data),
            "agents_profiling_data": {
                k: asdict(v) for k, v in self.agents_profiling_data.items()
            },
        }

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> State:
        """Deserialize a state from a dict (inverse of :meth:`to_dict`)."""
        state = State(**dictionary)
        agent_configs: dict[AgentType, AgentConfig] = {}
        answers = []
        for agent_type in state.agent_configs:
            agent_configs[agent_type] = AgentConfig.from_dict(
                dictionary["agent_configs"][agent_type]
            )
        for answer in dictionary["answers"]:
            answers.append(Answer.from_dict(answer))
        params = dict(dictionary)
        params["agent_configs"] = MappingProxyType(agent_configs)
        params["answers"] = answers
        return State(**params)

    def save(self, save_dir: Path):
        """Save the serialized state to a JSON file.

        The file is named ``<run_id>.json`` inside *save_dir*.
        """
        logger.info(f"Saving state to {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / f"{self.run_id}.json"
        with open(save_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


__all__ = ["State"]
