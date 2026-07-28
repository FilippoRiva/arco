import logging

from arco.core import Answer, Evaluation, Evaluator, State

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

    def extract_gt_from_answer(self, answer: Answer) -> dict:
        choice = answer.agent_output.get("agent_choice", "")
        plan = answer.agent_output.get("plan", [])
        return {"choice": choice.lower(), "plan": plan}


__all__ = ["PlannerEvaluator"]
