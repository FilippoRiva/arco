import dataclasses
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .agent_type import AgentType
from .answer import Answer
from .config import AgentConfig
from .profiling_data import ProfilingData


# Immutable dataclass representing the state
@dataclass(frozen=True)
class State:
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
        """
        Returns a new State object with ad added answer to the answers attribute
        :param answer: The answer to add to the answer's list
        :return: A new state object containing a new answer
        """
        return dataclasses.replace(self, answers=self.answers + [answer])

    def get_last_answer(self, agent_type: AgentType | None = None) -> Answer | None:
        """
        Retrieve the most recent answer entry for a specific agent type from the state.

        This method filters the 'answers' list by the class name of the provided
        agent type and returns the final occurrence.

        Args:
            agent_type: The class of the agent to search for (e.g., AgentType.VISUALIZER).
                The function uses `agent_type.__name__` to match against `agent_id`.

        Returns:
            The last answer produced by the agent of type agent_type
        """
        answers = self.answers
        if agent_type:
            return next((item for item in reversed(answers) if item.agent_id == agent_type), None)
        return answers[-1] if len(answers) > 0 else None

    def replace_last_answer(self, answer: Answer) -> State:
        last_answer = self.get_last_answer()
        if not last_answer:
            return dataclasses.replace(self, answers=[answer])
        new_answers = [answer.copy() for answer in self.answers]
        new_answers.pop(-1)
        new_answers.append(answer)
        return dataclasses.replace(self, answers=new_answers)

    def get_last_agent_config(self, agent_type: AgentType | None = None) -> AgentConfig | None:
        if not agent_type:
            la = self.get_last_answer()
            if not la: return None
            agent_type = la.agent_id
        return self.get_agent_config(agent_type)

    def get_agent_config(self, agent_type: AgentType) -> AgentConfig:
        if agent_type not in self.agent_configs.keys():
            raise Exception(
                f"The specified agent type ({agent_type}) is not defined in the {AgentType.__name__} enum. Please provide a known agent_type")
        return self.agent_configs[agent_type]

    def get_agents_used(self) -> list[str]:
        return [answer.agent_id.value.lower() for answer in self.answers if
                answer.agent_id is not AgentType.ORCHESTRATOR]

    def get_last_execution_outputs(self) -> tuple[Answer | None, AgentConfig | None]:
        return self.get_last_answer(), self.get_last_agent_config()

    def stringify_answers(self, max_message_length=100):
        """
        Converts the list of answer dictionaries into a single formatted string.

        Each entry is formatted as '(agent_id: message)', with entries separated
        by commas. This is typically used for logging or feeding the
        conversation history back into an LLM prompt.

        Returns:
            A comma-delimited string of all agent responses.
        """
        answers = self.answers
        return ",".join([
            f"({a.agent_id.value}: {a.message[:(max_message_length - 3)] + '...' if len(a.message) > max_message_length else a.message})"
            for a in answers])

    def set_profiling_data(self, profiling_data: ProfilingData, agent_type: AgentType) -> State:
        # Global level profiling
        global_profiling_data = self.global_profiling_data + profiling_data

        # Agent level profiling
        agents_profiling_data = self.agents_profiling_data.copy()
        if agent_type in self.agents_profiling_data:
            agents_profiling_data[agent_type] = agents_profiling_data[agent_type] + profiling_data
        else:
            agents_profiling_data[agent_type] = profiling_data

        # Answer level profiling
        self.get_last_answer(agent_type).profiling_data = profiling_data

        return replace(
            self,
            global_profiling_data=global_profiling_data,
            agents_profiling_data=agents_profiling_data
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> State:
        state = State(**dictionary)
        # Convert dicts back to AgentConfigs, Answers and Dict[AgentType,Answer]
        agent_configs = {}
        answers = []
        for agent_type in AgentType:
            if agent_type.value in state.agent_configs.keys():
                agent_configs[agent_type] = AgentConfig.from_dict(dictionary['agent_configs'][agent_type.value])
        for answer in dictionary['answers']:
            answers.append(Answer.from_dict(answer))
        dictionary.update({
            'agent_configs': agent_configs,
            'answers': answers,
        })
        return State(**dictionary)

    def save(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / f"{self.run_id}.json"
        with open(save_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
