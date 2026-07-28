import logging

from arco.core import Answer, Evaluation, Evaluator, State

logger = logging.getLogger(__name__)


class OrchestratorEvaluator(Evaluator):
    def _eval(self, state: State, judge_provider: str, judge_model: str):
        pass

    def _batch_eval(self, states: list[State]) -> bool:
        return False

    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        if answer.agent_output["agent_choice"].lower() == gt_data["choice"]:
            score = 1
        else:
            score = 0
        answer.gt_evaluation = Evaluation(score=score)
        logger.debug(f"Evaluation successful : score={score}")

    def extract_gt_from_answer(self, answer: Answer) -> dict:
        choice = answer.agent_output.get("agent_choice", "")
        return {"choice": choice}


__all__ = ["OrchestratorEvaluator"]
