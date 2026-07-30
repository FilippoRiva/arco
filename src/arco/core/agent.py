import difflib
import inspect
import logging
import math
import sys
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from . import llm_tools
from .agent_type import AgentType
from .evaluator import Evaluator
from .exceptions import AgentException
from .llm_tools import LLMAnswer
from .profiling_data import ProfilingData
from .state import State

if TYPE_CHECKING:
    from .config import AgentConfig
    from .llm_tools import LLM
    from .tracking import LLMCallAccumulator

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Abstract base class for all agents.

    Every concrete subclass is automatically registered in the
    :class:`AgentType` registry via :meth:`__init_subclass__`.

    Subclasses must implement :meth:`core` and may optionally override
    :meth:`post_generation_hooks` and the :attr:`evaluator` property.
    """

    def __init_subclass__(cls, **kwargs):
        """Register concrete subclasses in the AgentType registry.

        Intermediate abstract subclasses (those that still have abstract
        methods) are skipped.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        AgentType.register(cls.__name__)

    def __init__(self):
        self.type = AgentType(self.__class__.__name__)

    @property
    def name(self) -> str:
        """Return the agent type name (e.g. ``"Retriever"``)."""
        return self.type.value

    @property
    def evaluator(self) -> Evaluator | None:
        """Return the evaluator for best-of-N selection and GT evaluation.

        Subclasses should override this to return a specialized evaluator.
        ``None`` skips best-of-N evaluation.
        """
        return None

    @evaluator.setter
    def evaluator(self, evaluator: Evaluator):
        if isinstance(evaluator, Evaluator):
            return evaluator
        return self.evaluator

    @abstractmethod
    def core(self, state: State, llm: LLM) -> State:
        """Implement the agent's core logic.

        This is the only method a subclass must implement. It receives the
        current :class:`State` and an :class:`LLM` instance and returns an
        updated state with the agent's output appended.

        :param state: The current workflow state.
        :param llm: The LLM instance to use for inference.
        :returns: Updated :class:`State` with a new :class:`Answer` appended.
        """
        ...

    def answer(
        self,
        state: State,
        *,
        message: str = "",
        output: dict | None = None,
        error: str | None = None,
        logprobs=None,
    ) -> State:
        """Build an :class:`Answer` and append it to *state*.

        The answer's ``agent_id`` and ``agent_config`` are filled in
        automatically from the agent's type and the state's config.

        :param state: The current state.
        :param message: Human-readable summary of the agent's output.
        :param output: Structured output for downstream agents.
        :param error: Error message if the agent failed.
        :param logprobs: Token-level log probabilities from the LLM.
        :returns: A new state with the answer appended.
        """
        from .answer import Answer

        return state.add_answer(
            Answer(
                agent_id=self.type,
                agent_config=state.get_agent_config(self.type),
                message=message,
                agent_output=output or {},
                error=error,
                logprobs=logprobs,
            )
        )

    def post_generation_hooks(
        self, results: list[State], llm_acc: LLMCallAccumulator, config: AgentConfig
    ) -> list[State]:
        """Post-process best-of-N candidates before evaluation.

        Override this in subclasses to apply transformations across all
        candidates (e.g. column name standardisation in the Retriever).

        :param results: The list of candidate states from greedy or best-of-N execution.
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
            if state.get_last_answer(self.type).budget_controller_choice == "end":
                return state

    def _get_config_and_execute(self, state: State) -> State:
        """Resolve the agent config, run inference (greedy or best-of-N),
        apply post-generation hooks, evaluate best-of-N candidates, and
        attach profiling data.

        This is the core execution pipeline for a single agent step,
        called by :meth:`_invoke` on each iteration of the budget controller
        loop.

        :param state: The current workflow state.
        :returns: A new state with the best answer appended and profiling data attached.
        """
        agent_config: AgentConfig = state.get_agent_config(self.type)

        # Start timers
        agent_t0 = time.perf_counter()

        # Get llm call time accumulator for profiling
        from .llm_tools import LLMCallAccumulator

        llm_acc = LLMCallAccumulator(self.type)

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
                results=results, config=agent_config
            )
        else:
            best_result = results[0]

        ###
        # Profiling
        ###
        total_agent_time = time.perf_counter() - agent_t0
        profiling_data = ProfilingData(
            total_time=total_agent_time,
            llm_time=llm_acc.total_time,
            **llm_acc.energy_dict,
        )
        best_result = best_result.set_profiling_data(profiling_data, self.type)

        return best_result

    def _arco_evaluation(self, state: State) -> State:
        answer = state.get_last_answer(self.type)
        if not answer or answer.logprobs is None:
            return state

        # Compute Perplexity
        numeric_logprobs: list[float | int] = [probs for _, probs in answer.logprobs]
        avg_logprob = sum(numeric_logprobs) / len(numeric_logprobs)
        if avg_logprob < -math.log(sys.float_info.max):
            perplexity = math.inf
        else:
            perplexity = math.exp(-avg_logprob)

        answer.perplexity = perplexity
        return state.replace_last_answer(answer)

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

        max_perplexity = _AGENT_MAX_PERPLEXITY.get(self.type.value.lower()) or 2

        if answer.perplexity is not None and answer.perplexity > max_perplexity:
            answer.budget_controller_choice = "rollback"

            agent_config = state.get_agent_config(self.type)
            agent_config.temp_min = agent_config.temp_min * 0.9
            agent_config.temp_max = agent_config.temp_max * 0.95
            if agent_config.n < 3:
                agent_config.n = agent_config.n + 1

            return state

        answer.budget_controller_choice = "end"
        return state

    def _execute_greedy(
        self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator
    ) -> list[State]:
        # Instantiate LLM
        logger.debug("Starting greedy execution")
        llm = llm_tools.get_llm_from_config(agent_config=config, llm_acc=llm_acc)

        # Run inference
        result: State = self.core(state, llm)
        if config.cot_n > 1:
            result: State = self._apply_cot_iteration(
                state=state, llm=llm, max_iter=config.cot_n
            )
        return [result]

    def _execute_best_of_n(
        self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator
    ) -> list[State]:
        # Initialize results and their scores
        logger.debug("Starting best-of-n execution")
        results = []

        if config.provider is None or config.model is None:
            raise AgentException("Both config provider and config model must be set")

        # Generate results
        for i, (temp, top_p, top_k) in enumerate(config.get_candidate_params()):
            llm = llm_tools.get_llm(
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
            )

            result: State = self.core(state, llm)
            if config.cot_n > 1:
                result: State = self._apply_cot_iteration(state, llm, result, config)
            results.append(result)

        logger.debug(f"Best-of-n execution completed with {len(results)} candidates")
        return results

    def _apply_cot_iteration(self, state: State, llm: LLM, max_iter: int) -> State:
        _COT_SIMILARITY_THRESHOLD = 0.95

        llm.cot_enabled = True
        for cot_i in range(1, max_iter):
            # Apply Refinement
            llm.execution_error = state.answers[-1].error

            previous_output: LLMAnswer = llm.last_answer
            state: State = self.core(state, llm)
            current_output: LLMAnswer = llm.last_answer
            current_error = state.answers[-1].error

            if not current_error:
                ratio = difflib.SequenceMatcher(
                    None, previous_output, current_output
                ).ratio()

                if ratio >= self._COT_SIMILARITY_THRESHOLD:
                    break

        return state


__all__ = ["Agent"]
