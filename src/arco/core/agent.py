from __future__ import annotations

import difflib
import logging
import math
import sys
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .agent_type import AgentType
from .exceptions import AgentException
from .profiling_data import ProfilingData

if TYPE_CHECKING:
    from .answer import AnswerDraft
    from .config import AgentConfig
    from .evaluator import Evaluator
    from .llm_tools import LLM, LLMAnswer
    from .state import State
    from .tracking import LLMCallAccumulator

logger = logging.getLogger(__name__)
_ACTIVE_LLM_ACCUMULATOR: ContextVar[Any | None] = ContextVar(
    "arco_active_llm_accumulator", default=None
)


class Agent(ABC):
    """Abstract base class for all agents.

    Every concrete subclass is automatically registered in the
    :class:`AgentType` registry via :meth:`__init_subclass__`.

    Subclasses must implement :meth:`core` and may optionally override
    :meth:`post_generation_hooks` and the :attr:`evaluator` property.
    """

    def __init__(self, agent_name: str | AgentType | None = None):
        """Initialize an agent with an optional distinct workflow name.

        Most concrete agents use their class name. Generic agent classes that
        are instantiated more than once can provide ``agent_name`` so their
        answers and graph nodes remain distinguishable while their underlying
        agent type remains the same class.
        """
        self.type = AgentType(agent_name or self.__class__.__name__)

    @property
    def name(self) -> str:
        """Return the agent type name (e.g. ``"Retriever"``)."""
        return self.type

    def emit_event(
        self,
        event: str,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a custom LangGraph stream event.

        Event emission is best-effort so agents can still be invoked directly
        outside a LangGraph stream, such as in tests or notebooks.

        :param event: Application-defined event kind, such as ``"state"`` or
            ``"tool_started"``.
        :param message: Optional human-readable message.
        :param data: Optional JSON-serializable event payload.
        """
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except (ImportError, RuntimeError):
            return

        payload: dict[str, Any] = {
            "event": "agent_progress",
            "agent": str(self.name),
            "kind": event,
        }
        if message is not None:
            payload["message"] = message
        if data is not None:
            payload["data"] = data

        writer(payload)

    @property
    def evaluator(self) -> Evaluator | None:
        """Return the evaluator for best-of-N selection and GT evaluation.

        Subclasses should override this to return a specialized evaluator.
        ``None`` skips best-of-N evaluation.
        """
        return None

    @abstractmethod
    def core(self, state: State, llm: LLM) -> AnswerDraft:
        """Produce the agent's answer content.

        Subclasses implement this method and return an :class:`AnswerDraft`.
        The framework wraps the draft in a complete :class:`Answer`, attaching
        agent configuration and execution metadata before adding it to *state*.

        :param state: The current workflow state.
        :param llm: The LLM instance to use for inference.
        :returns: The agent-produced answer content and structured output.
        """
        ...

    def _run_core_and_populate_answer(self, state: State, llm: LLM) -> State:
        """Run ``core`` and let the framework finalize its answer draft."""
        from .answer import Answer, AnswerDraft

        draft = self.core(state, llm)
        if not isinstance(draft, AnswerDraft):
            raise TypeError(
                f"{self.__class__.__name__}.core() must return AnswerDraft; "
                f"received {type(draft).__name__}"
            )

        llm_acc = _ACTIVE_LLM_ACCUMULATOR.get()
        logprobs = getattr(llm_acc, "last_logprobs", []) if llm_acc else []
        thinking = getattr(llm_acc, "last_reasoning", None) if llm_acc else None
        answer = Answer(
            agent_id=self.type,
            agent_config=state.get_agent_config(self.type),
            message=draft.message,
            agent_output=draft.output,
            error=draft.error,
            logprobs=logprobs,
            thinking=thinking,
        )
        return state.add_answer(answer)

    def post_generation_hooks(
        self, results: list[State], llm_acc: LLMCallAccumulator, config: AgentConfig
    ) -> list[State]:
        """Post-process best-of-N candidates before evaluation.

        Override this in subclasses to apply transformations across all
        candidates (e.g. column name standardisation in the Retriever).

        :param results: The list of candidate states from greedy or best-of-N
        execution.
        :param llm_acc: The LLM call accumulator for this step.
        :param config: The agent's configuration for this execution.
        :returns: The (possibly modified) list of candidate states.
        """
        return results

    def __call__(self, state: State) -> State:
        return self._invoke(state)

    def _invoke(self, state: State) -> State:
        while True:
            state = self._get_config_and_execute(state)
            state = self._arco_evaluation(state)
            state = self._budget_controller(state)
            last_answer = state.get_last_answer(self.type)
            if (
                last_answer is not None
                and last_answer.budget_controller_choice == "end"
            ):
                return state

    def _get_config_and_execute(self, state: State) -> State:
        """Resolve the agent config, run inference (greedy or best-of-N),
        apply post-generation hooks, evaluate best-of-N candidates, and
        attach profiling data.

        This is the core execution pipeline for a single agent step,
        called by :meth:`_invoke` on each iteration of the budget controller
        loop.

        :param state: The current workflow state.
        :returns: A new state with the best answer appended and
        profiling data attached.
        """
        agent_config: AgentConfig = state.get_agent_config(self.type)

        # Start timers
        agent_t0 = time.perf_counter()

        # Get llm call time accumulator for profiling
        from .tracking import LLMCallAccumulator

        llm_acc = LLMCallAccumulator(self.type)
        # Start before agent logic so DB access, parsing, code execution, and
        # agents that do not call an LLM still receive energy profiling data.
        llm_acc.start()
        active_accumulator_token = _ACTIVE_LLM_ACCUMULATOR.set(llm_acc)

        try:
            ###
            # Inference
            ###
            if agent_config.n == 1:
                results = self._execute_greedy(
                    state=state, config=agent_config, llm_acc=llm_acc
                )
            else:
                results = self._execute_best_of_n(
                    state=state, config=agent_config, llm_acc=llm_acc
                )

            # Run Post Generation Hooks (dynamically overridden if needed, see Retriever as an example)
            results = self.post_generation_hooks(
                results, llm_acc=llm_acc, config=agent_config
            )

            ###
            # Evaluation
            ###
            if self.evaluator:
                results, best_result = self.evaluator.evaluate_best_of_n(
                    results=results,
                    config=agent_config,
                    llm_accumulator=llm_acc,
                )
            else:
                best_result = results[0]
        finally:
            # CodeCarbon is scoped to the complete agent step, including any
            # best-of-N judge calls, rather than to every individual LLM call.
            try:
                llm_acc.finish()
            finally:
                _ACTIVE_LLM_ACCUMULATOR.reset(active_accumulator_token)

        ###
        # Profiling
        ###
        total_agent_time = time.perf_counter() - agent_t0
        profiling_data = ProfilingData(
            total_time=total_agent_time,
            llm_time=llm_acc.total_time,
            **llm_acc.energy_dict,
        )
        logger.debug(
            f"Logging {self.type} profiling data. Codecarbon dict: {llm_acc.energy_dict}"
        )
        best_result = best_result.set_profiling_data(profiling_data, self.type)
        answer = best_result.get_last_answer(self.type)
        if answer is not None:
            usage = llm_acc.token_usage
            token_counts = {
                "input_token_count": usage.input_tokens,
                "output_token_count": usage.output_tokens,
                "total_token_count": usage.total_tokens,
                "cache_creation_token_count": usage.cache_creation_tokens,
                "cache_read_token_count": usage.cache_read_tokens,
                "reasoning_token_count": usage.reasoning_tokens,
            }
            for key, count in token_counts.items():
                existing_count = getattr(answer, key)
                if count is not None and existing_count is not None:
                    token_counts[key] = max(count, existing_count)
            token_counts = {
                key: count for key, count in token_counts.items() if count is not None
            }
            if token_counts:
                best_result = best_result.replace_last_answer(
                    answer.set(**token_counts)
                )

        return best_result

    def _arco_evaluation(self, state: State) -> State:
        answer = state.get_last_answer(self.type)
        if not answer or len(answer.logprobs) == 0:
            return state

        # Compute Perplexity
        numeric_logprobs: list[float | int] = [probs for _, probs in answer.logprobs]
        avg_logprob = sum(numeric_logprobs) / len(numeric_logprobs)
        if avg_logprob < -math.log(sys.float_info.max):
            perplexity = math.inf
        else:
            perplexity = math.exp(-avg_logprob)

        return state.replace_last_answer(answer.set(perplexity=perplexity))

    def _budget_controller(self, state: State) -> State:
        _AGENT_MAX_PERPLEXITY: dict[str, float] = {
            "retriever": 2,
            "analyzer": 15,
            "visualizer": 3,
            "orchestrator": 1.3,
            "planner": 1.3,
        }

        answer = state.get_last_answer(self.type)
        if not answer:
            return state

        max_perplexity = _AGENT_MAX_PERPLEXITY.get(self.type.lower()) or 2

        if answer.perplexity is not None and answer.perplexity > max_perplexity:
            choice = "rollback"
            state = state.replace_last_answer(
                answer.set(budget_controller_choice=choice)
            )

            agent_config = state.get_agent_config(self.type)
            new_n = agent_config.n + 1 if agent_config.n < 3 else agent_config.n
            new_config = agent_config.set(
                temp_min=agent_config.temp_min * 0.9,
                temp_max=agent_config.temp_max * 0.95,
                n=new_n,
            )
            new_configs = dict(state.agent_configs)
            new_configs[self.type] = new_config
            return dataclass_replace(state, agent_configs=MappingProxyType(new_configs))

        choice = "end"
        return state.replace_last_answer(answer.set(budget_controller_choice=choice))

    def _execute_greedy(
        self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator
    ) -> list[State]:
        # Instantiate LLM
        logger.debug("Starting greedy execution")
        from .llm_tools import get_llm_from_config

        llm = get_llm_from_config(agent_config=config, llm_acc=llm_acc)

        # Run inference
        llm_acc.reset_response_metadata()
        result: State = self._run_core_and_populate_answer(state, llm)
        if config.iterative_refinement_n > 1:
            result = self._apply_iterative_refinement(
                state=result,
                llm=llm,
                max_iter=config.iterative_refinement_n,
            )
        return [result]

    def _execute_best_of_n(
        self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator
    ) -> list[State]:
        # Initialize results and their scores
        logger.debug("Starting best-of-n execution")
        from .llm_tools import get_llm

        results = []

        if config.provider is None or config.model is None:
            raise AgentException("Both config provider and config model must be set")

        # Generate results
        for i, (temp, top_p, top_k) in enumerate(config.get_candidate_params()):
            llm = get_llm(
                # Variable
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                # Fixed
                max_tokens=config.max_tokens,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                llm_accumulator=llm_acc,
                provider=config.provider,
                model=config.model,
                enable_reasoning=bool(config.enable_reasoning),
                reasoning_effort=config.reasoning_effort,
                reasoning_summary=config.reasoning_summary,
                reasoning_max_tokens=config.reasoning_max_tokens,
                verbosity=config.verbosity,
                enable_logprobs=bool(config.enable_logprobs),
            )

            llm_acc.reset_response_metadata()
            result: State = self._run_core_and_populate_answer(state, llm)
            if config.iterative_refinement_n > 1:
                result = self._apply_iterative_refinement(
                    state=result,
                    llm=llm,
                    max_iter=config.iterative_refinement_n,
                )
            last_answer = result.get_last_answer(self.type)
            if last_answer is not None:
                result = result.replace_last_answer(
                    last_answer.set(
                        generation_params=(temp, top_p, top_k),
                    )
                )
            results.append(result)

        logger.debug(f"Best-of-n execution completed with {len(results)} candidates")
        return results

    def _apply_iterative_refinement(
        self, state: State, llm: LLM, max_iter: int
    ) -> State:
        """Iteratively refine an already-generated agent result."""
        similarity_threshold = 0.95

        llm.iterative_refinement_enabled = True
        loop_state: State = state
        for _ in range(1, max_iter):
            previous_output: LLMAnswer | None = llm.last_answer
            previous_answer = loop_state.get_last_answer(self.type)
            llm.execution_error = previous_answer.error if previous_answer else None

            active_accumulator = _ACTIVE_LLM_ACCUMULATOR.get()
            if active_accumulator is not None:
                active_accumulator.reset_response_metadata()
            loop_state = self._run_core_and_populate_answer(loop_state, llm)
            current_output: LLMAnswer | None = llm.last_answer
            current_answer = loop_state.get_last_answer(self.type)
            current_error = current_answer.error if current_answer else None

            if (
                previous_output is not None
                and current_output is not None
                and not current_error
            ):
                ratio = difflib.SequenceMatcher(
                    None, previous_output.text, current_output.text
                ).ratio()

                if ratio >= similarity_threshold:
                    break

        return loop_state


__all__ = ["Agent"]
