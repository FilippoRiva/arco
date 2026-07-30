import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .profiling_data import ProfilingData

if TYPE_CHECKING:
    from ..data.benchmark_dataset import BenchmarkEntry, BenchmarkSummary
    from . import AgentConfig, AgentType, Answer, State

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
        """Deserialize from a dict.

        :param dictionary: Dict with keys ``"score"`` and ``"success"``.
        :returns: A new :class:`Evaluation` instance.
        """
        return Evaluation(
            score=float(dictionary["score"]), success=bool(dictionary["success"])
        )


class Evaluator(ABC):
    """Abstract base class for agent evaluation strategies.

    Subclasses must implement :meth:`_eval`, :meth:`_batch_eval`, and
    :meth:`_gt_eval`.  The public methods :meth:`evaluate_best_of_n` and
    :meth:`evaluate_ground_truth` orchestrate the evaluation flow.
    """

    def evaluate_best_of_n(
        self, results: list[State], config: AgentConfig
    ) -> tuple[list[State], State]:
        """Evaluate a list of candidate states and select the best one.

        Tries :meth:`_batch_eval` first (e.g. pairwise IoU for retrievers).
        If it returns ``False``, falls back to :meth:`_eval` for each
        candidate individually (e.g. LLM-as-a-judge).

        :param results: The candidate states from greedy or best-of-N execution.
        :param config: The agent's configuration (used to get judge model).
        :returns: A tuple ``(all_results, best_result)`` where *best_result*
            is the selected state with discarded answers attached.
        """
        if len(results) == 1:
            return results, results[0]

        logger.debug("Evaluating best n results")

        batch_eval_success = self._batch_eval(results)
        if not batch_eval_success:
            for result in results:
                self._eval(
                    result,
                    judge_provider=config.provider_judge,
                    judge_model=config.model_judge,
                )

        return results, Evaluator._selection(results)

    def evaluate_ground_truth(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        """Run ground-truth evaluation on a single answer.

        Delegates to :meth:`_gt_eval` and stores the result in
        ``answer.gt_evaluation``.

        :param answer: The answer to evaluate.
        :param gt_data: Ground-truth data dict (structure depends on the
            evaluator subclass).
        :param judge_provider: Provider for the LLM judge.
        :param judge_model: Model for the LLM judge.
        """
        logger.info(
            f"Evaluating ground truth data for {answer.agent_id} with this data : {gt_data}"
        )
        self._gt_eval(
            answer=answer,
            gt_data=gt_data,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )

    @staticmethod
    def _selection(states: list[State]) -> State:
        """Select the best state from a list based on evaluation scores.

        Prefers states with a successful evaluation and picks the highest
        score.  Discarded candidates are attached to the selected state.
        """
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
    def _eval(self, state: State, judge_provider: str, judge_model: str):
        """Evaluate a single candidate state.

        Called by :meth:`evaluate_best_of_n` when :meth:`_batch_eval` fails.
        Implementations should set ``state.get_last_answer().evaluation`` to
        an :class:`Evaluation` object.

        :param state: A candidate state containing at least one agent answer.
        :param judge_provider: The LLM provider to use for judging.
        :param judge_model: The LLM model to use for judging.
        """
        ...

    @abstractmethod
    def _batch_eval(self, states: list[State]) -> bool:
        """Evaluate all candidates together (batch comparison).

        Called by :meth:`evaluate_best_of_n` before falling back to
        :meth:`_eval`.  Implementations should set ``.evaluation`` on
        each candidate's answer.

        :param states: All candidate states from best-of-N execution.
        :returns: ``True`` if batch evaluation succeeded (no per-candidate
            fallback needed), ``False`` otherwise.
        """
        ...

    @abstractmethod
    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        """Evaluate a single answer against ground-truth data.

        Called by :meth:`evaluate_ground_truth`.  Implementations should
        set ``answer.gt_evaluation`` to an :class:`Evaluation` object.

        :param answer: The answer to evaluate.
        :param gt_data: Ground-truth data dict.  The expected keys depend
            on the evaluator subclass (e.g. ``"choice"`` for orchestrator,
            ``"analysis"`` for analyzer, ``"chart_config"`` for visualizer).
        :param judge_provider: The LLM provider to use for judging.
        :param judge_model: The LLM model to use for judging.
        """
        ...

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
    evaluators: dict[AgentType, Evaluator],
    judge_provider: str,
    judge_model: str,
) -> BenchmarkSummary:
    """Compare a workflow's execution trace against a ground-truth benchmark entry.

    Walks through the state's answers in order, matching each against the
    expected trace element.  Stops at the first divergence.  For each
    matching step, runs ground-truth evaluation and collects metrics.

    :param state: The final state produced by the workflow.
    :param entry: The ground-truth :class:`BenchmarkEntry` to compare against.
    :param evaluators: Dict mapping agent types to their evaluators.
    :param judge_provider: Provider for the LLM judge.
    :param judge_model: Model for the LLM judge.
    :returns: A :class:`BenchmarkSummary` with completion rate, scores,
        perplexities, and profiling data.
    """
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
