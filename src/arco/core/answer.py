import io
from dataclasses import dataclass, field, asdict
from typing import List, Any

import pandas as pd
from pandas import DataFrame

from .agent import AgentType
from .agent_config import AgentConfig
from .evaluator import Evaluation
from .profiling_data import ProfilingData


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
    logprobs: list[tuple[str, float | int]] | None = None
    perplexity: float = 0.0

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
        if ans.data_str:
            ans.data_df = pd.read_csv(io.StringIO(ans.data_str))
        if ans.profiling_data:
            ans.profiling_data = ProfilingData(**dictionary["profiling_data"])
        return ans

    def copy(self) -> Answer:
        return Answer.from_dict(self.to_dict())
