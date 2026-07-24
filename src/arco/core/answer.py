from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .agent import AgentType
from .agent_config import AgentConfig
from .evaluator import Evaluation
from .profiling_data import ProfilingData


@dataclass
class Answer:
    # Main Model
    agent_id: AgentType
    message: str  # for visualization purposes
    agent_config: AgentConfig
    agent_output: dict = field(
        default_factory=defaultdict(lambda: None))  # whatever the agent outputs is put here for other agents to access

    # Evaluation
    evaluation: Evaluation | None = None
    gt_evaluation: Evaluation | None = None

    # Discarded Best-of-N Answers
    discarded_bon_answers: list[Answer] | None = None

    # Error message
    error: str | None = None

    # LLM generation info
    logprobs: list[tuple[str, float | int]] | None = None
    perplexity: float | None = None

    # Profiling Data
    profiling_data: ProfilingData = field(default_factory=ProfilingData)

    # ARCO info
    budget_controller_choice: Literal["rollback", "end"] = "end"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> Answer:
        ans = Answer(**dictionary)
        ans.agent_config = AgentConfig.from_dict(dictionary["agent_config"])
        if ans.agent_id:
            ans.agent_id = AgentType(ans.agent_id)
        if ans.evaluation:
            ans.evaluation = Evaluation.from_dict(dictionary['evaluation'])
        if ans.gt_evaluation:
            ans.gt_evaluation = Evaluation.from_dict(dictionary['gt_evaluation'])
        if ans.discarded_bon_answers:
            ans.discarded_bon_answers = [
                Answer.from_dict(discarded_ans) for discarded_ans in dictionary['discarded_bon_answers']
            ]
        if ans.profiling_data:
            ans.profiling_data = ProfilingData(**dictionary["profiling_data"])
        return ans

    def copy(self) -> Answer:
        return deepcopy(self)
