from arco.core import Answer, Evaluation, Evaluator


class OrchestratorEvaluator(Evaluator):
    def _gt_eval(self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str):
        if answer.agent_output['agent_choice'].lower() == gt_data['choice']:
            answer.gt_evaluation = Evaluation(score=1)
        else:
            answer.gt_evaluation = Evaluation(score=0)
