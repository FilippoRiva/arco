import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .profiling_data import ProfilingData

if TYPE_CHECKING:
    from ..data.benchmark_dataset import BenchmarkEntry, BenchmarkSummary
    from . import AgentConfig, Answer, State

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Result of a single evaluation (best-of-N or ground-truth).

    :ivar score: Normalised score in ``[0, 1]``.
    :ivar success: Whether the evaluation passed (default ``True``).
    """

    score: float
    success: bool = True

    @classmethod
    def from_dict(cls, dictionary: dict) -> Evaluation:
        return Evaluation(
            score=float(dictionary["score"]), success=bool(dictionary["success"])
        )


class Evaluator(ABC):
    """Abstract base class for agent evaluation strategies.

    Subclasses must implement :meth:`_eval`, :meth:`_batch_eval`, and
    :meth:`_gt_eval`.  All three return :class:`Evaluation` objects
    instead of mutating answers.
    """

    def evaluate_best_of_n(
        self, results: list[State], config: AgentConfig
    ) -> tuple[list[State], State]:
        """Evaluate candidates and select the best one.

        Tries :meth:`_batch_eval` first.  If it returns ``None``,
        falls back to :meth:`_eval` per candidate.

        :param results: Candidate states from greedy or best-of-N execution.
        :param config: Agent configuration (used for judge model).
        :returns: ``(all_results, best_result, evaluations)``.
        """
        if len(results) == 1:
            return results, results[0]

        logger.debug("Evaluating best n results")

        # evaluate results
        evaluations = self._batch_eval(results)
        if evaluations is None:
            evaluations = []
            for result in results:
                ev = self._eval(
                    result,
                    judge_provider=config.provider_judge,
                    judge_model=config.model_judge,
                )
                evaluations.append(ev)

        # update results with their evaluations
        new_results = []
        for result, evaluation in zip(results, evaluations):
            last_answer = result.get_last_answer()
            new_results.append(
                result.replace_last_answer(last_answer.set(evaluation=evaluation))
            )
        results = new_results

        # select the best result
        best_state = Evaluator._selection(results, evaluations)

        return new_results, best_state

    def evaluate_ground_truth(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ) -> Evaluation:
        """Run ground-truth evaluation and return the result.

        :param answer: The answer to evaluate.
        :param gt_data: Ground-truth data dict.
        :param judge_provider: Provider for the LLM judge.
        :param judge_model: Model for the LLM judge.
        :returns: An :class:`Evaluation` with the result.
        """
        logger.info(
            f"Evaluating ground truth data for {answer.agent_id} with this data : {gt_data}"
        )
        return self._gt_eval(
            answer=answer,
            gt_data=gt_data,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )

    @staticmethod
    def _selection(states: list[State], evaluations: list[Evaluation]) -> State:
        """Select the best state given its evaluations.

        Discarded candidates are attached to the selected state's answer.
        """
        if any(e.score != 0 for e in evaluations):
            idx = max(range(len(evaluations)), key=lambda i: evaluations[i].score)
        else:
            idx = 0
        best_state = states[idx]

        discarded_answers = []
        for i, s in enumerate(states):
            if i != idx:
                ans = s.get_last_answer()
                if ans is not None:
                    discarded_answers.append(ans)

        best_answer = best_state.get_last_answer()
        if best_answer is not None:
            from dataclasses import replace as dc_replace

            new_answer = dc_replace(
                best_answer, discarded_bon_answers=discarded_answers
            )
            best_state = best_state.replace_last_answer(new_answer)

        return best_state

    @abstractmethod
    def _eval(
        self, state: State, judge_provider: str, judge_model: str
    ) -> Evaluation: ...

    @abstractmethod
    def _batch_eval(self, states: list[State]) -> list[Evaluation] | None: ...

    @abstractmethod
    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ) -> Evaluation: ...

    def extract_gt_from_answer(self, answer: Answer) -> dict:
        """Extract ground-truth data from an Answer for benchmark generation.

        Subclasses override this to return the dict that :meth:`_gt_eval`
        expects.  The default returns an empty dict.

        :param answer: The answer to extract data from.
        :returns: A dict suitable for storage in a benchmark trace entry.
        """
        return {}


def evaluate_state_with_benchmark_entry(
    state: State,
    entry: BenchmarkEntry,
    evaluators: dict[str, Evaluator],
    judge_provider: str,
    judge_model: str,
) -> tuple[BenchmarkSummary, list[Evaluation | None]]:
    """Compare a workflow's execution trace against a ground-truth benchmark entry.

    Walks through the state's answers in order, matching each against the
    expected trace element.  Stops at the first divergence.  For each
    matching step, runs ground-truth evaluation and collects metrics.

    :param state: The final state produced by the workflow.
    :param entry: The ground-truth :class:`BenchmarkEntry` to compare against.
    :param evaluators: Dict mapping agent type names to their evaluators.
    :param judge_provider: Provider for the LLM judge.
    :param judge_model: Model for the LLM judge.
    :returns: A tuple ``(summary, evaluations)`` where *evaluations* has
        one entry per answer in the state (``None`` for answers beyond the
        first trace divergence).
    """
    from arco.data import BenchmarkSummary

    correct_path = 0
    ppls: list[float] = []
    scores: list[float] = []
    agents: list[str] = []
    profiling_datas: list[ProfilingData] = []
    all_evaluations: list[Evaluation | None] = []

    for idx, answer in enumerate(state.answers):
        if idx > len(entry.trace) - 1:
            all_evaluations.append(None)
            continue
        correct_trace = entry.trace[idx]
        if answer.agent_id == correct_trace.agent_type:
            correct_path += 1
        else:
            all_evaluations.append(None)
            continue

        evaluator = evaluators.get(answer.agent_id)
        if evaluator is not None:
            evaluation = evaluator.evaluate_ground_truth(
                answer=answer,
                gt_data=correct_trace.data,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
            all_evaluations.append(evaluation)
            ppls.append(answer.perplexity)
            scores.append(evaluation.score)
            agents.append(answer.agent_id)
            profiling_datas.append(answer.profiling_data)
        else:
            all_evaluations.append(None)

    completion_percentage = correct_path / len(entry.trace) if entry.trace else 0.0

    summary = BenchmarkSummary(
        completion_percentage=completion_percentage,
        ppls=ppls,
        scores=scores,
        agents=agents,
        profiling_datas=profiling_datas,
    )
    return summary, all_evaluations


__all__ = ["Evaluation", "Evaluator", "evaluate_state_with_benchmark_entry"]
