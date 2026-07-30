import dataclasses
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .agent_type import AgentType
from .answer import Answer
from .config import AgentConfig
from .profiling_data import ProfilingData

logger = logging.getLogger(__name__)


# Immutable dataclass representing the state
@dataclass(frozen=True)
class State:
    """Immutable workflow state propagated through the LangGraph.

    :ivar prompt: The original user prompt.
    :ivar run_id: Unique identifier for this run.
    :ivar agent_configs: Per-agent configuration dict.
    :ivar answers: Ordered list of agent answers produced so far.
    :ivar global_profiling_data: Cumulative profiling data across all steps.
    :ivar agents_profiling_data: Per-agent cumulative profiling data.
    """

    # Original prompt
    prompt: str

    # Run unique identifier
    run_id: str

    # Dynamic Configuration for agents
    agent_configs: dict[AgentType, AgentConfig]

    # List of agent's answers
    answers: list[Answer] = field(default_factory=list)

    # List of metrics profiling the current state
    global_profiling_data: ProfilingData = field(default_factory=ProfilingData)
    agents_profiling_data: dict[AgentType, ProfilingData] = field(default_factory=dict)

    def add_answer(self, answer: Answer) -> State:
        """Return a new state with *answer* appended to the answers list.

        Args:
            answer: The answer to append.

        Returns:
            A new state with the answer added.
        """
        return dataclasses.replace(self, answers=self.answers + [answer])

    def get_last_answer(self, agent_type: AgentType | None = None) -> Answer | None:
        """Retrieve the most recent answer for a specific agent type.

        Args:
            agent_type: The agent type to filter by (e.g. ``AgentType.VISUALIZER``).
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
        new_answers = [answer.copy() for answer in self.answers]
        new_answers.pop(-1)
        new_answers.append(answer)
        return dataclasses.replace(self, answers=new_answers)

    def get_last_agent_config(
        self, agent_type: AgentType | None = None
    ) -> AgentConfig | None:
        """Return the config for the agent that produced the last answer."""
        if not agent_type:
            la = self.get_last_answer()
            if not la:
                return None
            agent_type = la.agent_id
        return self.get_agent_config(agent_type)

    def get_agent_config(self, agent_type: AgentType) -> AgentConfig:
        """Return the config for a given agent type."""
        return self.agent_configs[agent_type]

    def get_agents_used(self) -> list[str]:
        """Return the list of agent type names used so far, excluding orchestrator."""
        return [
            answer.agent_id.value.lower()
            for answer in self.answers
            if answer.agent_id is not AgentType.ORCHESTRATOR
        ]

    def set_profiling_data(
        self, profiling_data: ProfilingData, agent_type: AgentType
    ) -> State:
        """Attach profiling data and return a new state.

        Accumulates into ``global_profiling_data``, per-agent
        ``agents_profiling_data``, and the last answer's profiling data.
        """
        global_profiling_data = self.global_profiling_data + profiling_data

        agents_profiling_data = self.agents_profiling_data.copy()
        if agent_type in self.agents_profiling_data:
            agents_profiling_data[agent_type] = (
                agents_profiling_data[agent_type] + profiling_data
            )
        else:
            agents_profiling_data[agent_type] = profiling_data

        self.get_last_answer(agent_type).profiling_data = profiling_data

        return replace(
            self,
            global_profiling_data=global_profiling_data,
            agents_profiling_data=agents_profiling_data,
        )

    def to_dict(self) -> dict:
        """Serialize the state to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> State:
        """Deserialize a state from a dict (inverse of :meth:`to_dict`)."""
        state = State(**dictionary)
        agent_configs = {}
        answers = []
        for agent_type in AgentType.all():
            if agent_type.value in state.agent_configs:
                agent_configs[agent_type] = AgentConfig.from_dict(
                    dictionary["agent_configs"][agent_type.value]
                )
        for answer in dictionary["answers"]:
            answers.append(Answer.from_dict(answer))
        dictionary.update(
            {
                "agent_configs": agent_configs,
                "answers": answers,
            }
        )
        return State(**dictionary)

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
