import json
from typing import Optional, Dict, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco.core import AgentType, State, Answer, Evaluation, Evaluator
from arco.core.llm_tools import get_llm


class OrchestratorEvaluator(Evaluator):
    def _gt_eval(self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str):
        if answer.agent_choice.lower() == gt_data['choice']:
            answer.gt_evaluation=Evaluation(score=1)
        else:
            answer.gt_evaluation=Evaluation(score=0)
