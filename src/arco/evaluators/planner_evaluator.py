import logging

from arco.core import Answer, Evaluation, Evaluator, State

logger = logging.getLogger(__name__)


class PlannerEvaluator(Evaluator):
    def _batch_eval(self, states: list[State]) -> list[Evaluation] | None:
        return None

    def _eval(self, state: State, judge_provider: str, judge_model: str) -> Evaluation:
        return Evaluation(score=0.0)

    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ) -> Evaluation:
        gen_choice = answer.agent_output.get("agent_choice", "").lower()
        expected_choice = gt_data.get("choice", "").lower()
        score = 1.0 if gen_choice == expected_choice else 0.0
        logger.debug(f"Evaluation successful : score={score}")
        return Evaluation(score=score)

    def extract_gt_from_answer(self, answer: Answer) -> dict:
        choice = answer.agent_output.get("agent_choice", "")
        return {"choice": choice}


__all__ = ["PlannerEvaluator"]
