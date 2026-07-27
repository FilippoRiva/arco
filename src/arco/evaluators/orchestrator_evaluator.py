from typing import TYPE_CHECKING

from arco.core import Evaluation, Evaluator

if TYPE_CHECKING:
    from arco.core import Answer


class OrchestratorEvaluator(Evaluator):
    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        if answer.agent_output["agent_choice"].lower() == gt_data["choice"]:
            answer.gt_evaluation = Evaluation(score=1)
        else:
            answer.gt_evaluation = Evaluation(score=0)


__all__ = ["OrchestratorEvaluator"]
