from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from .profiling_data import ProfilingData

if TYPE_CHECKING:
    from ..data.benchmark_dataset import BenchmarkEntry, BenchmarkSummary
    from .state import State
    from . import Answer, State, AgentType, AgentConfig


@dataclass(frozen=True)
class Evaluation:
    score: float
    success: bool = True

    @classmethod
    def from_dict(cls, dictionary: dict):
        return Evaluation(
            score=float(dictionary['score']),
            success=bool(dictionary['success']))


class Evaluator:
    def evaluate_and_select(self, results: List[State], config: AgentConfig) -> Tuple[List[State], State]:
        if len(results) == 1:
            return results, results[0]

        # executes _batch_eval, if that fails it runs _eval
        batch_eval_success = self._batch_eval(results)
        if not batch_eval_success:
            for result in results:
                self._eval(result, judge_provider=config.provider_judge, judge_model=config.model_judge)

        # finally selects the best result
        return results, self._selection(results)

    def _eval(self, state: State, judge_provider: str, judge_model: str):
        la = state.get_last_answer()
        if not la:
            raise ValueError("Tried to evaluate a State with no Answers")

        la.evaluation = Evaluation(score=0.0, success=False)
        return

    def _batch_eval(self, states: List[State]) -> bool:
        answers_with_none: List[Answer | None] = [state.get_last_answer() for state in states]
        answers = [ans for ans in answers_with_none if ans is not None]
        for answer in answers:
            answer.gt_evaluation = Evaluation(score=0.0, success=False)
        return False  # default implementation has no success

    # noinspection PyMethodMayBeStatic
    def _selection(self, states: List[State]) -> State:
        answers_with_none: List[Answer | None] = [state.get_last_answer() for state in states]
        answers = [ans for ans in answers_with_none if ans is not None]
        if any(answer.evaluation is None for answer in answers):
            return states[0]
        if any(answer.evaluation.success == False for answer in answers):  # pyrefly: ignore [missing-attribute]
            return states[0]
        best_state = max(states, key=lambda r: r.get_last_answer().evaluation.score)
        discarded_states = [*states]
        discarded_states.remove(best_state)
        best_state.get_last_answer().discarded_bon_answers = [state.get_last_answer() for state in discarded_states]
        return best_state

    def evaluate_ground_truth(self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str):
        """Run ground-truth evaluation for tracking/logging only."""
        self._gt_eval(answer=answer, gt_data=gt_data, judge_provider=judge_provider, judge_model=judge_model)

    def _gt_eval(self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str):
        answer.gt_evaluation = Evaluation(score=0.0, success=False)
        return


def evaluate_state(state: State, entry: BenchmarkEntry, evaluators: dict[AgentType, Evaluator], judge_provider: str,
                   judge_model: str) -> BenchmarkSummary:
    correct_path = 0
    ppls: list[float] = []
    scores: list[float] = []
    agents: list[AgentType] = []
    profiling_datas: list[ProfilingData] = []
    for idx, answer in enumerate(state.answers):
        correct_trace = entry.trace[idx]
        if answer.agent_id == correct_trace.agent_type:
            correct_path += 1
        else:
            break

        evaluators[answer.agent_id].evaluate_ground_truth(answer=answer, gt_data=correct_trace.data,
                                                          judge_provider=judge_provider, judge_model=judge_model)
        evaluation = answer.gt_evaluation
        ppls.append(answer.perplexity)
        scores.append(evaluation.score)
        agents.append(answer.agent_id)
        profiling_datas.append(answer.profiling_data)
    completion_percentage = correct_path / len(entry.trace)
    from arco.data import BenchmarkSummary
    return BenchmarkSummary(
        completion_percentage=completion_percentage,
        ppls=ppls,
        scores=scores,
        agents=agents,
        profiling_datas=profiling_datas
    )
