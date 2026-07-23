from arco.core import Answer, Evaluation, Evaluator


class PlannerEvaluator(Evaluator):
    def _gt_eval(self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str):
        gen_choice = answer.agent_output.get("agent_choice", "").lower()
        expected_choice = gt_data.get("choice", "").lower()
        if gen_choice == expected_choice:
            answer.gt_evaluation = Evaluation(score=1)
        else:
            answer.gt_evaluation = Evaluation(score=0)