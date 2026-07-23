import difflib
import math
import sys
import time
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from . import llm_tools
from .agent_type import AgentType
from .evaluator import Evaluator
from .exceptions import AgentException
from .profiling_data import ProfilingData
from .state import State

if TYPE_CHECKING:
    from arco.core.llm_tools import CoTRefiner
    from arco.core.tracking import LLMCallAccumulator
    from .config import AgentConfig


class Agent(ABC):
    _COT_SIMILARITY_THRESHOLD = 0.95

    def __init__(self):
        self.type = AgentType(self.__class__.__name__)

    @property
    def name(self) -> str:
        return self.type.value

    @abstractmethod
    def core(self, state: State, llm: BaseChatModel | CoTRefiner) -> State:
        """
            Provides the core functionality of the agent.

            Args:
                state (State): State of the agent.
                llm : BaseChatModel instance of a large language model to be used at inference

            Returns:
                Updated state with analysis appended to answers
        """
        ...

    def execute_greedy(self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator) -> List[State]:
        # Instantiate LLM
        llm = llm_tools.get_llm_from_config(agent_config=config, llm_acc=llm_acc)

        # Run inference
        result: State = self.core(state, llm)
        if config.cot_n > 1:
            result: State = self.apply_cot_iteration(
                state,
                llm,
                result,
                config
            )
        return [result]

    def execute_best_of_n(self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator) -> List[State]:
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
                result: State = self.apply_cot_iteration(
                    state,
                    llm,
                    result,
                    config
                )
            results.append(result)
        return results

    def apply_cot_iteration(
            self,
            state: State,
            llm,
            initial_result: State,
            config: AgentConfig
    ) -> State:
        """Apply up to cot_n iterative CoT refinement steps to a single LLM call.

        Starting from initial_result, repeatedly re-invokes core() with a
        CoTRefinementLLM that appends the previous response to every prompt.
        Stops early when the output converges (similarity >= _COT_SIMILARITY_THRESHOLD)
        or after cot_n total iterations (including the initial one).

        Args:
            state: Current agent state (unchanged across iterations).
            llm: The base LLM instance (temperature already set by the caller).
            initial_result: Result from the first (non-refinement) call.
            config: AgentConfig used to get

        Returns:
            The result from the final (or converged) iteration.
        """

        result = initial_result
        previous_output: str = str(initial_result.answers[-1])
        execution_error: str | None = initial_result.answers[-1].error

        for cot_i in range(1, config.cot_n):
            # Apply Refinement
            refinement_llm = llm_tools.CoTRefiner(llm, previous_output, execution_error)
            new_result: State = self.core(state, refinement_llm)

            new_error = new_result.answers[-1].error
            if new_error:
                # If an error is found it continues until the error is fixed or max iterations is reached
                previous_output = str(new_result.answers[-1])
                execution_error = new_error

                result = new_result
            else:
                new_output = str(new_result.answers[-1])
                ratio = difflib.SequenceMatcher(None, previous_output, new_output).ratio()

                if ratio >= self._COT_SIMILARITY_THRESHOLD:
                    break

                previous_output = new_output

                result = new_result

        return result

    def arco_evaluation(self, state: State) -> State:
        # Fetch the last answer object using your parent node identifier
        answer = state.get_last_answer(self.type)
        if not answer:
            raise Exception("No answer found during ARCO evaluation")
        if answer.logprobs is None or len(answer.logprobs) == 0:
            answer.perplexity = 0.0
            return state.replace_last_answer(answer)

        # Compute Perplexity
        numeric_logprobs: list[float | int] = [probs for _, probs in answer.logprobs]
        avg_logprob = sum(numeric_logprobs) / len(numeric_logprobs)
        if avg_logprob < -math.log(sys.float_info.max):
            perplexity = math.inf
        else:
            perplexity = math.exp(-avg_logprob)

        answer.perplexity = perplexity
        return state.replace_last_answer(answer)

    def budget_controller(self, state: State) -> State:
        answer = state.get_last_answer(self.type)
        if not answer:
            raise Exception("No answer found during budget controller phase")

        if self.type == AgentType.RETRIEVER:
            max_perplexity = 2
        elif self.type == AgentType.ANALYZER:
            max_perplexity = 15
        elif self.type == AgentType.VISUALIZER:
            max_perplexity = 3
        elif self.type == AgentType.ORCHESTRATOR:
            max_perplexity = 1.3
        else:
            raise Exception(
                f"The Budget Controller does not implement answer evaluation for this type of Agent : {parent_node.value}")

        if answer.perplexity > max_perplexity:
            answer.budget_controller_choice = "rollback"

            agent_config = state.get_agent_config(self.type)
            agent_config.temp_min = agent_config.temp_min * 0.9
            agent_config.temp_max = agent_config.temp_max * 0.95
            if agent_config.n < 3:
                agent_config.n = agent_config.n + 1

            return state

        answer.budget_controller_choice = "end"
        return state

    def post_generation_hooks(self, results: List[State], llm_acc: LLMCallAccumulator, config: AgentConfig) -> List[
        State]:
        return results

    def get_config_and_execute(self, state: State) -> State:
        """Execute a step with per-step best-of-n, evaluation, and caching.
        Args:
            state: Current agent state
        Returns:
            The new resulting State of the best execution
        """

        agent_config: AgentConfig = state.get_agent_config(self.type)

        # Start timers
        agent_t0 = time.perf_counter()

        # Get llm call time accumulator for profiling
        from arco.core.llm_tools import LLMCallAccumulator
        llm_acc = LLMCallAccumulator(self.type)

        ###
        # Inference
        ###
        if agent_config.n == 1:
            results = self.execute_greedy(state=state, config=agent_config, llm_acc=llm_acc)
        else:
            results = self.execute_best_of_n(state=state, config=agent_config, llm_acc=llm_acc)

        # Run Post Generation Hooks (dynamically overridden if needed, see Retriever as an example)
        results = self.post_generation_hooks(results, llm_acc=llm_acc, config=agent_config)

        ###
        # Evaluation
        ###
        results, best_result = self.get_evaluator().evaluate_and_select(results=results, config=agent_config)

        ###
        # Profiling
        ###
        total_agent_time = time.perf_counter() - agent_t0
        profiling_data = ProfilingData(total_time=total_agent_time,
                                       llm_time=llm_acc.total_time,
                                       **llm_acc.energy_dict)
        best_result = best_result.set_profiling_data(profiling_data, self.type)

        return best_result

    def invoke(self, state: State) -> State:
        while True:
            state = self.get_config_and_execute(state)
            state = self.arco_evaluation(state)
            state = self.budget_controller(state)
            if state.get_last_answer(self.type).budget_controller_choice == "end":
                return state

    @staticmethod
    def get_evaluator() -> Evaluator:
        return Evaluator()

    def __call__(self, state: State) -> State:
        return self.invoke(state)
