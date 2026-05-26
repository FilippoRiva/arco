from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from .exceptions import EvaluatorException

if TYPE_CHECKING:
    from .state import State, Answer


@dataclass(frozen=True)
class Evaluation:
    score: float
    success: bool = True


class Evaluator:
    def evaluate_and_select(self, results: List[State]) -> Tuple[List[State], State]:
        # executes _batch_eval, if that fails it runs _eval
        batch_eval_success = self._batch_eval(results)
        if not batch_eval_success:
            for result in results:
                self._eval(result)
        # finally selects the best result
        return results, self._selection(results)

    def _eval(self, state: State):
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
        if any(answer.evaluation.success == False for answer in answers): # pyrefly: ignore [missing-attribute]
            return states[0]
        best_state = max(states, key=lambda r: r.get_last_answer().evaluation.score)
        discarded_states = [*states]
        discarded_states.remove(best_state)
        # pyrefly: ignore [bad-assignment, missing-attribute]
        best_state.get_last_answer().discarded_bon_answers = [state.get_last_answer() for state in discarded_states]
        return best_state

    def evaluate_ground_truth(self, results: List[State]):
        """Run ground-truth evaluation for tracking/logging only.
        This NEVER influences selection — it only logs GT scores on the
        results so performance can be tracked without steering the agent.
        """
        for result in results:
            self._gt_eval(result)

    def _gt_eval(self, state: State):
        la : Answer | None = state.get_last_answer()
        if la is not None:
            la.gt_evaluation = Evaluation(score=0.0, success=False)
        return
