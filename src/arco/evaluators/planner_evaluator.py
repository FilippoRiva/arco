from typing import TYPE_CHECKING

from arco.core import Evaluation, Evaluator, State

if TYPE_CHECKING:
    from arco.core import Answer

import logging

logger = logging.getLogger(__name__)


class PlannerEvaluator(Evaluator):
    def _batch_eval(self, states: list[State]) -> bool:
        return False

    def _eval(self, state: State, judge_provider: str, judge_model: str):
        pass

    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        gen_choice = answer.agent_output.get("agent_choice", "").lower()
        expected_choice = gt_data.get("choice", "").lower()
        if gen_choice == expected_choice:
            score = 1
        else:
            score = 0

        answer.gt_evaluation = Evaluation(score=score)
        logger.debug(f"Evaluation successful : score={score}")


__all__ = ["PlannerEvaluator"]
