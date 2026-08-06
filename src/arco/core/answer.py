from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal, Self

from .agent import AgentType
from .agent_config import AgentConfig
from .evaluator import Evaluation
from .profiling_data import ProfilingData


@dataclass(frozen=True, slots=True)
class Answer:
    """Container for a single agent's output within a workflow run.

    Stores the agent's structured output, evaluation results, profiling
    data, and metadata about the generation (logprobs, perplexity, etc.).

    :ivar agent_id: The type of agent that produced this answer.
    :ivar message: Human-readable summary of the agent's output (for
          visualization purposes).
    :ivar agent_config: The configuration used for this agent's execution.
    :ivar agent_output: Structured output dict consumed by downstream
          agents (e.g. ``{"code": "...", "chart_config": {...}}``).
    :ivar evaluation: Best-of-N evaluation result (set during execution).
    :ivar gt_evaluation: Ground-truth evaluation result (set during
          benchmark evaluation).
    :ivar discarded_bon_answers: Best-of-N candidates that were not
          selected (recursive list of :class:`Answer` objects).
    :ivar error: Error message if the agent's execution failed.
    :ivar logprobs: Token-level log probabilities from the LLM.
    :ivar perplexity: Perplexity computed from the logprobs.
    :ivar profiling_data: Timing and energy profiling data for this step.
    :ivar budget_controller_choice: Whether the budget controller decided
          to accept (``"end"``) or re-execute (``"rollback"``).
    """

    agent_id: AgentType
    message: str
    agent_config: AgentConfig
    agent_output: dict = field(default_factory=dict)
    evaluation: Evaluation | None = None
    gt_evaluation: Evaluation | None = None
    discarded_bon_answers: list[Self] | None = None
    error: str | None = None
    logprobs: list[tuple[str, float | int]] | None = None
    perplexity: float | None = None
    profiling_data: ProfilingData = field(default_factory=ProfilingData)
    budget_controller_choice: Literal["rollback", "end"] = "end"

    def to_dict(self) -> dict:
        """Serialize the answer to a JSON-compatible dict.

        :returns: A dict with all fields converted to plain Python types.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> Self:
        """Deserialize an answer from a dict (inverse of :meth:`to_dict`).

        Recursively converts nested dicts back into :class:`AgentConfig`,
        :class:`Evaluation`, :class:`AgentType`, and :class:`Answer` objects.

        :param dictionary: The dict produced by :meth:`to_dict`.
        :returns: A new :class:`Answer` instance.
        """
        dictionary["agent_config"] = AgentConfig.from_dict(dictionary["agent_config"])
        if "agent_id" in dictionary:
            dictionary["agent_id"] = AgentType(dictionary["agent_id"])
        if dictionary.get("evaluation") is not None:
            dictionary["evaluation"] = Evaluation.from_dict(dictionary["evaluation"])
        if dictionary.get("gt_evaluation") is not None:
            dictionary["gt_evaluation"] = Evaluation.from_dict(
                dictionary["gt_evaluation"]
            )
        if dictionary.get("discarded_bon_answers") is not None:
            dictionary["discarded_bon_answers"] = [
                Answer.from_dict(discarded_ans)
                for discarded_ans in dictionary["discarded_bon_answers"]
            ]
        if dictionary.get("profiling_data") is not None:
            dictionary["profiling_data"] = ProfilingData(**dictionary["profiling_data"])
        return cls(**dictionary)

    def set(self, **kwargs) -> Self:
        """Return a new Answer with the given fields replaced.

        :param kwargs: Field names and their new values.
        :returns: A new :class:`Answer` instance.
        """
        valid_keys = {f.name for f in fields(Answer)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return replace(self, **filtered)

    def copy(self) -> Self:
        """Return a deep copy of this answer."""
        return deepcopy(self)
