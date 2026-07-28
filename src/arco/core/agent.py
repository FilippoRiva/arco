import difflib
import inspect
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


class Agent(ABC):
    def __init_subclass__(cls, **kwargs):
        """When a subclass inherits this ABC, the agent_type of that subclass is stored and the AgentType "dynamic ENUM".
        This provides compatibility with any kind of dynamically defined Agent"""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # don't register intermediate abstract subclasses
        AgentType.register(cls.__name__)

    def __init__(self):
        self.type = AgentType(self.__class__.__name__)

    @property
    def name(self) -> str:
        return self.type.value

    @property
    def evaluator(self) -> Evaluator | None:
        return None

    @evaluator.setter
    def evaluator(self, evaluator: Evaluator):
        if isinstance(evaluator, Evaluator):
            return evaluator
        return self.evaluator

    @abstractmethod
    def core(self, state: State, llm: LLM) -> State:
        """
        Provides the core functionality of the agent.

        Args:
            state (State): State of the agent.
            llm : LLM instance of a large language model to be used at inference

        Returns:
            Updated state with analysis appended to answers
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
        """Build an :class:`Answer` with implicit *agent_id* and *agent_config*, and append it to *state*."""
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
