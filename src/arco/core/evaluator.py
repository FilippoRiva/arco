import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .profiling_data import ProfilingData

if TYPE_CHECKING:
    from ..data.benchmark_dataset import BenchmarkEntry, BenchmarkSummary
    from . import AgentConfig, AgentType, Answer, State

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Evaluation:
    score: float
    success: bool = True

    @classmethod
    def from_dict(cls, dictionary: dict):
        return Evaluation(
            score=float(dictionary["score"]), success=bool(dictionary["success"])
        )


class Evaluator(ABC):
    def evaluate_best_of_n(
        self, results: list[State], config: AgentConfig
    ) -> tuple[list[State], State]:
        if len(results) == 1:
            return results, results[0]

        logger.debug("Evaluating best n results")

        # executes _batch_eval, if that fails it runs _eval
        batch_eval_success = self._batch_eval(results)
        if not batch_eval_success:
            for result in results:
                self._eval(
                    result,
                    judge_provider=config.provider_judge,
                    judge_model=config.model_judge,
                )

        # finally selects the best result
        return results, Evaluator._selection(results)

    def evaluate_ground_truth(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        logger.info(
            f"Evaluating ground truth data for {answer.agent_id} with this data : {gt_data}"
        )
        """Run ground-truth evaluation for tracking/logging only."""
        self._gt_eval(
            answer=answer,
            gt_data=gt_data,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )

    @staticmethod
    def _selection(states: list[State]) -> State:
        answers_with_none: list[Answer | None] = [
            state.get_last_answer() for state in states
        ]
        answers = [ans for ans in answers_with_none if ans is not None]
        if any(answer.evaluation is None for answer in answers):
            return states[0]
        if any(answer.evaluation.success == False for answer in answers):  # pyrefly: ignore [missing-attribute]
            return states[0]
        best_state = max(states, key=lambda r: r.get_last_answer().evaluation.score)
        discarded_states = [*states]
        discarded_states.remove(best_state)
        best_state.get_last_answer().discarded_bon_answers = [
            state.get_last_answer() for state in discarded_states
        ]
        return best_state

    @abstractmethod
    def _eval(self, state: State, judge_provider: str, judge_model: str): ...

    @abstractmethod
    def _batch_eval(self, states: list[State]) -> bool: ...

    @abstractmethod
    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ): ...

    def extract_gt_from_answer(self, answer: Answer) -> dict:
        """Extract ground-truth data from an Answer for benchmark generation.

        Subclasses override this to return the dict that ``_gt_eval`` expects.
        The default returns an empty dict.
        """
        return {}


def evaluate_state_with_benchmark_entry(
    state: State,
    entry: BenchmarkEntry,
    evaluators: dict[AgentType, Evaluator],
    judge_provider: str,
    judge_model: str,
) -> BenchmarkSummary:
    correct_path = 0
    ppls: list[float] = []
    scores: list[float] = []
    agents: list[AgentType] = []
    profiling_datas: list[ProfilingData] = []
    for idx, answer in enumerate(state.answers):
        if idx > len(entry.trace) - 1:
            break
        correct_trace = entry.trace[idx]
        if answer.agent_id == correct_trace.agent_type:
            correct_path += 1
        else:
            break

        evaluator = evaluators[answer.agent_id]
        if evaluator is not None:
            evaluator.evaluate_ground_truth(
                answer=answer,
                gt_data=correct_trace.data,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
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
        profiling_datas=profiling_datas,
    )


__all__ = ["Evaluation", "Evaluator", "evaluate_state_with_benchmark_entry"]
