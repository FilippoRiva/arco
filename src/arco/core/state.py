from copy import deepcopy

from pandas import DataFrame
import dataclasses
import io
import time
from dataclasses import dataclass, field, replace, asdict
from enum import Enum
from typing import List, Optional, Any

import pandas as pd

from .config import AgentConfig
from .evaluator import Evaluation


class AgentType(str, Enum):
    RETRIEVER = "Retriever"
    ANALYZER = "Analyzer"
    VISUALIZER = "Visualizer"
    ORCHESTRATOR = "Orchestrator"
    NONE = None


@dataclass
class Answer:
    # Main Model
    agent_id: AgentType
    message: str
    agent_config: AgentConfig

    # Evaluation
    evaluation: Evaluation | None = None
    gt_evaluation: Evaluation | None = None

    # Orchestrator output
    agent_choice: str | None = None

    # Retriever output
    data_str: str | None = None
    data_df: DataFrame | None = None
    sql_query: str | None = None

    # Analyzer output
    analysis: str | None = None

    # Visualizer output
    chart_config: dict | None = None
    code: str | None = None

    # Discarded Best-of-N Answers
    discarded_bon_answers: List[Answer] | None = None

    # Error message
    error: str | None = None

    # LLM generation info
    logprobs : list[float | int] | None = None

    def __str__(self) -> str:
        """
        Formats the answer for LLM context consumption.
        Excludes empty fields and massive data objects to save tokens.
        """
        # Agent name and message
        lines = [f"### Agent: {self.agent_id}"]
        if self.message:
            lines.append(f"Message: {self.message}")

        # Specific information enclosed in the message
        if self.agent_choice:
            lines.append(f"Decision: Selected next agent -> {self.agent_choice}")

        if self.sql_query:
            lines.append(f"SQL Query Executed:\n```sql\n{self.sql_query}\n```")

        if self.analysis:
            lines.append(f"Data Analysis: {self.analysis}")

        if self.chart_config:
            lines.append(f"Chart Configuration: {self.chart_config}")

        if self.code:
            lines.append(f"Generated Python Code:\n```python\n{self.code}\n```")

        # Highlight errors
        if self.error:
            lines.append(f"!!! ERROR OCCURRED !!!\n{self.error}")

        return "\n".join(lines)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> Answer:
        ans = Answer(**dictionary)
        ans.agent_config = AgentConfig.from_dict(dictionary["agent_config"])
        if ans.agent_id:
            ans.agent_id = AgentType(ans.agent_id)
        if ans.evaluation:
            ans.evaluation = Evaluation(score=float(dictionary['evaluation']['score']),
                                        success=bool(dictionary['evaluation']['success']))
        if ans.gt_evaluation:
            ans.gt_evaluation = Evaluation(score=float(dictionary['gt_evaluation']['score']),
                                           success=bool(dictionary['gt_evaluation']['success']))
        if ans.discarded_bon_answers:
            ans.discarded_bon_answers = [
                Answer.from_dict(discarded_ans) for discarded_ans in dictionary['discarded_bon_answers']
            ]
        if ans.data_str:
            ans.data_df = pd.read_csv(io.StringIO(ans.data_str))
        return ans

    def copy(self)-> Answer:
        return Answer.from_dict(self.to_dict())


# Immutable dataclass representing the state
@dataclass(frozen=True)
class State:
    # Original prompt and visualization goal
    prompt: str
    visualization_goal: str

    # Run unique identifier
    run_id: str

    # Dynamic Configuration from agents
    agent_configs: dict[AgentType, AgentConfig]

    # List of agent's answers
    answers: List[Answer] = field(default_factory=list)

    # List of metrics profiling the current state
    profiling_metrics: dict[str, Any] = field(default_factory=dict)

    # Caching
    cached_results: Optional[dict[AgentType, Answer]] = None  # Preloaded results from similar past runs

    def add_answer(self, answer: Answer) -> State:
        """
        Returns a new State object with ad added answer to the answers attribute
        :param answer: The answer to add to the answer's list
        :return: A new state object containing a new answer
        """
        return dataclasses.replace(self, answers=self.answers + [answer])

    def get_last_answer(self, agent_type: Optional[AgentType] = None) -> Answer | None:
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

    def replace_last_answer(self, answer: Answer)-> State:
        last_answer = self.get_last_answer()
        if not last_answer:
            return dataclasses.replace(self, answers = [answer])
        new_answers = [answer.copy() for answer in self.answers]
        new_answers.pop(-1)
        new_answers.append(answer)
        return dataclasses.replace(self, answers = new_answers)

    def get_last_agent_config(self, agent_type: AgentType | None = None) -> AgentConfig | None:
        if not agent_type:
            la = self.get_last_answer()
            if not la: return None
            agent_type = la.agent_id
        return self.get_agent_config(agent_type)

    def get_agent_config(self, agent_type: AgentType) -> AgentConfig:
        if agent_type not in self.agent_configs.keys():
            Exception(f"The specified agent type ({agent_type}) is not defined in the {AgentType.__name__} enum. Please provide a known agent_type")
        return self.agent_configs[agent_type]

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

    def set_profiling_metrics(self, total_llm_timings: float, agent_t0: float, agent_type: AgentType,
                              energy: dict[str, float] | None = None) -> State:
        profiling_metrics: dict[str, dict[str, float | dict]] = {}

        # Profiling : total time
        agent_timing = time.perf_counter() - agent_t0
        profiling_metrics['agent_timings_sec'] = {
            agent_type.value: round(agent_timing, 3)
        }

        # Profiling : llm time
        profiling_metrics['agent_llm_timings_sec'] = {
            agent_type.value: round(total_llm_timings, 3)
        }

        # Profiling : codecarbon
        if energy:
            profiling_metrics['codecarbon_energy'] = {
                agent_type.value: energy
            }

        return replace(
            self,
            profiling_metrics=profiling_metrics
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> State:
        state = State(**dictionary)
        # Convert dicts back to AgentConfigs, Answers and Dict[AgentType,Answer]
        agent_configs = {}
        answers = []
        cached_results = None
        for agent_type in AgentType:
            if agent_type.value in state.agent_configs.keys():
                agent_configs[agent_type] = AgentConfig.from_dict(dictionary['agent_configs'][agent_type.value])
        for answer in dictionary['answers']:
            if 'perplexity' in answer.keys():
                from arco.core import EmpoweredAnswer
                answers.append(EmpoweredAnswer.from_dict(answer))
            else:
                answers.append(Answer.from_dict(answer))
        if dictionary['cached_results']:
            cached_results = {}
            for k, v in dictionary['cached_results'].items():
                if 'perplexity' in v.keys():
                    from arco.core import EmpoweredAnswer
                    cached_results[AgentType(k)] = EmpoweredAnswer.from_dict(v)
                else:
                    cached_results[AgentType(k)] = Answer.from_dict(v)
        dictionary.update({
            'agent_configs': agent_configs,
            'answers': answers,
            'cached_results': cached_results
        })
        return State(**dictionary)

