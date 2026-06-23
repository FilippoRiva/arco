from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from functools import partial

from langchain_core.runnables import Runnable

from arco import tracing
import difflib
import time
from typing import List, TYPE_CHECKING, Callable

from langchain_core.language_models import BaseChatModel

from arco import llm_tools
from arco.tracing import _summarize_for_trace, truncate_trace_text
from .evaluator import Evaluator
from .empower import empower
from .state import State, AgentType
from .exceptions import AgentException

if TYPE_CHECKING:
    from arco.tracing import TracingHelper
    from arco.llm_tools import LLMCallAccumulator, CoTRefiner
    from .config import AgentConfig

class Agent:
    _COT_SIMILARITY_THRESHOLD = 0.95

    def __init__(self, trace_helper: TracingHelper):
        self.trace_helper: TracingHelper = trace_helper
        self.type: AgentType = AgentType.NONE

    def core(self, state: State, llm: BaseChatModel | CoTRefiner) -> State:
        """
            Provides the core functionality of the agent.

            Args:
                state (State): State of the agent.
                llm : BaseChatModel instance of a large language model to be used at inference

            Returns:
                Updated state with analysis appended to answers
        """
        return state

    def execute_greedy(self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator) -> List[State]:
        # Instantiate LLM
        llm = llm_tools.get_llm_from_config(agent_config=config, llm_acc=llm_acc)

        # Run inference
        temp, top_p, top_k = config.get_candidate_params()[0]
        with self.trace_helper.start_span(
                "step_candidate",
                kind="tool",
                attributes={
                    "step_name": config.agent_name,
                    "candidate_index": 0,
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "llm.provider": config.provider,
                    "llm.model": config.model,
                },
        ) as candidate_span:
            result: State = self.core(state, llm)
            if config.cot_n > 1:
                result: State = self.apply_cot_iteration(
                    state,
                    llm,
                    result,
                    config
                )

            config_dictionary = {
                "_temperature": temp,
                "_top_p": top_p,
                "_top_k": top_k,
                "_run_idx": 0,
            }
            tracing.set_output(candidate_span,
                                         _summarize_for_trace(result, additional_logging=config_dictionary))

        return [result]

    def execute_best_of_n(self, state: State, config: AgentConfig, llm_acc: LLMCallAccumulator,
                          agent_span) -> List[State]:
        # Initialize results and their scores
        results = []

        # Variable LLM parameters
        _param_idx = {"temperature": 0, "top_p": 1, "top_k": 2}[config.bon_parameter]
        varying_vals = [p[_param_idx] for p in config.get_candidate_params()]
        tracing.set_attributes(agent_span, {"candidate_count": config.n, "varying_values": varying_vals})

        # Generate results
        for i, (temp, top_p, top_k) in enumerate(config.get_candidate_params()):
            if config.provider is None or config.model is None:
                raise AgentException("Both config provider and config model must be set")

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
                ollama_url=config.ollama_url,
            )

            varying_val = varying_vals[i]

            with self.trace_helper.start_span(
                    "agent_candidate",
                    kind="tool",
                    attributes={
                        "agent_name": config.agent_name,
                        "candidate_index": i,
                        "temperature": temp,
                        "top_p": top_p,
                        "top_k": top_k,
                        config.bon_parameter: varying_val,
                        "llm.provider": config.provider,
                        "llm.model": config.model,
                    },
            ) as candidate_span:
                result: State = self.core(state, llm)
                if config.cot_n > 1:
                    result: State = self.apply_cot_iteration(
                        state,
                        llm,
                        result,
                        config
                    )
                config_dictionary = {
                    "_temperature": temp,
                    "_top_p": top_p,
                    "_top_k": top_k,
                    "_bon_param": config.bon_parameter,
                    "_run_idx": i,
                }

                tracing.set_output(candidate_span,
                                             _summarize_for_trace(result, additional_logging=config_dictionary))
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
            with self.trace_helper.start_span(
                    "cot_refinement",
                    kind="tool",
                    attributes={
                        "agent_name": config.agent_name,
                        "cot_iteration": cot_i + 1,
                        "cot_total": config.cot_n,
                    },
                    input_data={
                        "previous_output": truncate_trace_text(previous_output),
                        "execution_error": truncate_trace_text(execution_error),
                    },
            ) as span:
                new_result: State = self.core(state, refinement_llm)
                tracing.set_output(span, _summarize_for_trace(new_result))

            new_error = new_result.answers[-1].error
            if new_error:
                # If an error is found it continues until the error is fixed or max iterations is reached
                previous_output = str(new_result.answers[-1])
                execution_error = new_error

                result = new_result
            else:
                new_output = str(new_result.answers[-1])
                ratio = difflib.SequenceMatcher(None, previous_output, new_output).ratio()
                tracing.set_attributes(
                    span,
                    {
                        "cot_similarity": ratio,
                        "cot_converged": ratio >= self._COT_SIMILARITY_THRESHOLD,
                    },
                )

                if ratio >= self._COT_SIMILARITY_THRESHOLD:
                    break

                previous_output = new_output

                result = new_result

        return result

    def post_generation_hooks(self, results: List[State], llm_acc: LLMCallAccumulator, config: AgentConfig) -> List[
        State]:
        return results

    def get_config_and_execute(self, state: State) -> State:
        """Execute a step with per-step best-of-n, evaluation, and caching.
        Args:
            state: Current agent state
        Returns:
            Updated state dict from the best run
        """

        agent_config: AgentConfig = state.get_agent_config(self.type)

        with self.trace_helper.start_span(
                self.__class__.__name__,
                kind="agent",
                attributes={
                    "agent_name": agent_config.agent_name,
                    "config.cache_mode": getattr(agent_config, "cache_mode", None),
                    "config.use_cache": getattr(agent_config, "use_cache", None),
                    "config.n": getattr(agent_config, "n", None),
                    "config.cot_n": getattr(agent_config, "cot_n", None),
                },
                input_data=_summarize_for_trace(state),
        ) as agent_span:
            # Loads results from the cache if requested and returns if it hits
            if agent_config.use_cache:
                if state.cached_results:
                    if answer := state.cached_results.get(self.type):
                        return state.add_answer(answer)

            # Parameters for per-step LLM call instrumentation
            tracing.set_attributes(
                agent_span,
                {
                    "config.n": agent_config.n,
                    "config.cot_n": agent_config.cot_n,
                    "config.bon_param": agent_config.bon_parameter,
                    "config.max_tokens": agent_config.max_tokens,
                },
            )
            # Start timers
            agent_t0 = time.perf_counter()

            # Get llm call time accumulator for profiling
            from arco.llm_tools import LLMCallAccumulator
            llm_acc = LLMCallAccumulator(agent_config.agent_name)

            ###
            # Inference
            ###
            if agent_config.n == 1:
                results = self.execute_greedy(state=state, config=agent_config, llm_acc=llm_acc)
            else:
                results = self.execute_best_of_n(state=state, config=agent_config, agent_span=agent_span,
                                                 llm_acc=llm_acc)

            # Run Post Generation Hooks (dynamically overridden if needed, see Retriever as an example)
            results = self.post_generation_hooks(results, llm_acc=llm_acc, config=agent_config)

            ###
            # Evaluation
            ###
            evaluator: Evaluator = self.get_evaluator(agent_config)
            results, best_result = evaluator.evaluate_and_select(results=results)

            # Evaluate from gt for tracing purposes
            if self.can_evaluate_from_gt(agent_config):
                evaluator.evaluate_ground_truth(results=results)

            ###
            # Profiling
            ###
            best_result = best_result.set_profiling_metrics(
                total_llm_timings=llm_acc.total_time,
                agent_t0=agent_t0,
                agent_type=self.type,
                energy=llm_acc.total_energy,
            )

            return best_result


    def get_evaluator(self, agent_config: AgentConfig) -> Evaluator:
        return Evaluator()

    def can_evaluate_from_gt(self, agent_config: AgentConfig) -> bool:
        return False

    def get_node(self) -> Runnable[State, State]:
        return empower(RunnableLambda(self.get_config_and_execute))
