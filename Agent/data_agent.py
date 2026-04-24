"""Sales Data Agent using LangGraph, DuckDB, and Ollama (LLaMA).

This module exposes a class `SalesDataAgent` that orchestrates:
- DuckDB SQL over a local parquet file
- LLM-driven tool routing (lookup → analyze → visualize)
- Chart configuration extraction and chart code generation

Usage example:
    from Agent.data_agent import SalesDataAgent

    agent = SalesDataAgent()
    result = agent.run("Show me the sales in Nov 2021")
    print(result["answer"])  # Ordered list of steps/outputs (analysis text, then code)
"""

import difflib
import requests
import json
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import tempfile
import numpy as np

import pandas as pd

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

try:
    from Agent.parameter_provider import ParameterProvider
    from Agent.utils import text_to_csv, save_csv, get_evaluation_functions, make_csv_evaluator_no_gt, make_text_evaluator_no_gt, make_vis_evaluator_no_gt
    from Agent.config import AgentConfig, StepConfig
    from Agent.cache import RunCache
    from Agent.schema import DatabaseSchema, TableSchema, ColumnSchema
    from Agent.tracing import (TracingHelper, _truncate_trace_text, _summarize_state_for_trace,
                               _summarize_result_for_trace, _PHOENIX_AVAILABLE,
                               phoenix_register, LangChainInstrumentor, _TRACE_LIST_LIMIT)
    from Agent.state import State, DEFAULT_DATA_PATH
    from Agent.steps import (lookup_sales_data_core, decide_tool_core, analyzing_data_core,
                             create_visualization_core, route_to_tool, _extract_step_output,
                             CoTRefinementLLM)
except ImportError:
    from parameter_provider import ParameterProvider
    from utils import text_to_csv, save_csv, get_evaluation_functions, make_csv_evaluator_no_gt, make_text_evaluator_no_gt, make_vis_evaluator_no_gt
    from config import AgentConfig, StepConfig
    from cache import RunCache
    from schema import DatabaseSchema, TableSchema, ColumnSchema
    from tracing import (TracingHelper, _truncate_trace_text, _summarize_state_for_trace,
                         _summarize_result_for_trace, _PHOENIX_AVAILABLE,
                         phoenix_register, LangChainInstrumentor, _TRACE_LIST_LIMIT)
    from state import State, DEFAULT_DATA_PATH
    from steps import (lookup_sales_data_core, decide_tool_core, analyzing_data_core,
                       create_visualization_core, route_to_tool, _extract_step_output,
                       CoTRefinementLLM)

_COT_SIMILARITY_THRESHOLD = 0.95


class _LLMCallAccumulator(BaseCallbackHandler):
    """Accumulates wall-clock time and energy of LLM .invoke() calls via LangChain callbacks.

    Attach as a callback to a LangChain LLM object to record only the time and energy
    spent inside actual LLM API calls, excluding non-LLM work (DB queries, parquet reads,
    code execution, etc.) that may be present in the same step function.

    When cc_enabled=True, a fresh CodeCarbon EmissionsTracker is started at the beginning
    of each invoke() and stopped at the end. This avoids the pro-rating approximation that
    would be incorrect when GPU power varies significantly during inference (e.g. local
    Ollama on A40/L40S), since the tracker window covers only the actual inference window.

    Thread-safe for sequential use (one step at a time).
    """

    def __init__(
        self,
        cc_enabled: bool = False,
        cc_output_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._starts: Dict[str, float] = {}
        self._cc_trackers: Dict[str, Any] = {}
        self.total_time: float = 0.0
        self._cc_enabled = cc_enabled and _CODECARBON_AVAILABLE
        self._cc_output_dir = cc_output_dir
        # Accumulated energy across all invoke() calls for this step
        self.total_energy: Dict[str, float] = {
            "energy_consumed_kwh": 0.0,
            "cpu_energy_kwh": 0.0,
            "gpu_energy_kwh": 0.0,
            "ram_energy_kwh": 0.0,
            "emissions_kg_co2": 0.0,
        }

    def _start_cc_tracker(self, key: str) -> None:
        if not self._cc_enabled:
            return
        try:
            tracker = EmissionsTracker(  # type: ignore[call-arg]
                project_name="llm_invoke",
                output_dir=self._cc_output_dir,
                save_to_file=False,
                measure_power_secs=1,
                log_level="error",
                allow_multiple_runs=True,
            )
            tracker.start()
            self._cc_trackers[key] = tracker
        except Exception as _e:
            print(f"[CodeCarbon] per-invoke tracker start failed: {_e}")

    def _stop_cc_tracker(self, key: str) -> None:
        tracker = self._cc_trackers.pop(key, None)
        if tracker is None:
            return
        try:
            tracker.stop()
            # tracker.stop() returns a float (CO2 kg), not EmissionsData.
            # The full breakdown is in final_emissions_data, same as the original code.
            _ed = getattr(tracker, "final_emissions_data", None)
            if _ed is not None:
                self.total_energy["energy_consumed_kwh"] += getattr(_ed, "energy_consumed", 0.0) or 0.0
                self.total_energy["cpu_energy_kwh"]      += getattr(_ed, "cpu_energy",      0.0) or 0.0
                self.total_energy["gpu_energy_kwh"]      += getattr(_ed, "gpu_energy",      0.0) or 0.0
                self.total_energy["ram_energy_kwh"]      += getattr(_ed, "ram_energy",      0.0) or 0.0
                self.total_energy["emissions_kg_co2"]    += getattr(_ed, "emissions",        0.0) or 0.0
        except Exception as _e:
            print(f"[CodeCarbon] per-invoke tracker stop failed: {_e}")

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        key = str(run_id)
        self._starts[key] = time.perf_counter()
        self._start_cc_tracker(key)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        key = str(run_id)
        if key in self._starts:
            self.total_time += time.perf_counter() - self._starts.pop(key)
        self._stop_cc_tracker(key)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        # Count errored calls too — the HTTP round-trip still happened.
        key = str(run_id)
        if key in self._starts:
            self.total_time += time.perf_counter() - self._starts.pop(key)
        self._stop_cc_tracker(key)

# Optional energy/emissions tracking via CodeCarbon
try:
    from codecarbon import EmissionsTracker  # type: ignore
    print("CodeCarbon is available")
    _CODECARBON_AVAILABLE = True
except Exception:
    print("CodeCarbon is not available, not using it")
    EmissionsTracker = None  # type: ignore
    _CODECARBON_AVAILABLE = False

# Mirror utils_0.py printing of langgraph version
import langgraph
import langgraph.version
print(langgraph.version)


class SalesDataAgent:
    """End-to-end agent to query, analyze, and visualize sales data.

    The agent builds a LangGraph with tool-selection, data lookup (DuckDB over
    parquet), LLM-based analysis, and visualization code generation. Use `run()`
    to execute a single prompt through the flow.
    """
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        streaming: bool = True,
        data_path: Optional[str] = None,
        schema: Optional["DatabaseSchema"] = None,
        ollama_url: Optional[str] = None,
        enable_tracing: bool = False,
        phoenix_api_key: Optional[str] = None,
        phoenix_endpoint: Optional[str] = None,
        project_name: str = "evaluating-agent",
        provider: str = "openai",
        openai_api_key: Optional[str] = None,
        # New: Per-step configuration and caching
        agent_config: Optional[AgentConfig] = None,
        cache_dir: Optional[str] = None,
        # Runtime parameter provider (default: static config from YAML)
        parameter_provider: Optional["ParameterProvider"] = None,
    ) -> None:
        """Initialize the agent and compile the graph.

        Args:
            model: Model name (OpenAI model like "gpt-4o-mini" or Ollama model like "llama3.2:3b").
            temperature: Sampling temperature for the LLM.
            max_tokens: Generation token limit.
            streaming: Whether to stream tokens from the LLM.
            data_path: Optional override for the parquet dataset path (single-table legacy mode).
            schema: Optional DatabaseSchema for multi-table support. When provided, takes
                    precedence over data_path for query execution. If None, auto-builds a
                    minimal schema from data_path at query time.
            ollama_url: Optional override for Ollama base URL; defaults to OLLAMA_HOST or http://localhost:11434.
            provider: LLM provider to use ("ollama" or "openai"). Default is "ollama".
            openai_api_key: Optional OpenAI API key; defaults to OPENAI_API_KEY env var.
            agent_config: Optional AgentConfig for per-step hyperparameter control.
            cache_dir: Optional directory for caching run results.
            parameter_provider: Optional ParameterProvider for runtime step config
                overrides. When None, defaults to DefaultProvider (static YAML config).
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided via openai_api_key parameter or OPENAI_API_KEY environment variable")
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                api_key=api_key,
            )
            self.ollama_url = None
        else:  # ollama
            self.ollama_url = ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                base_url=self.ollama_url,
            )

        self.data_path = data_path or DEFAULT_DATA_PATH
        # Multi-table schema. None means single-table legacy mode (auto-built at query time).
        self.schema = schema

        # Optional Phoenix/OpenInference tracing integration
        self.tracer = None
        self.tracing_enabled = False
        if enable_tracing and _PHOENIX_AVAILABLE:
            try:
                # Environment variables similar to utils_0.py
                if phoenix_api_key:
                    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
                    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"
                if phoenix_endpoint:
                    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

                tracer_provider = phoenix_register(
                    project_name=project_name,
                    endpoint=(phoenix_endpoint or "https://app.phoenix.arize.com/v1/traces"),
                )
                LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)
                self.tracer = tracer_provider.get_tracer(__name__)
                self.tracing_enabled = True
            except Exception as _:
                self.tracer = None
                self.tracing_enabled = False
        self.trace_helper = TracingHelper(self.tracer if self.tracing_enabled else None)

        # Store model parameters for LLM factory method
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Initialize per-step configuration
        if agent_config is not None:
            self.agent_config = agent_config
        else:
            # Create default config with current parameters
            self.agent_config = AgentConfig(
                model=model,
                provider=provider,
                ollama_url=self.ollama_url or "http://localhost:11434",
                openai_api_key=self.openai_api_key,
            )

        # Initialize runtime parameter provider
        from .parameter_provider import DefaultProvider
        self.parameter_provider = parameter_provider or DefaultProvider()

        # Initialize result cache
        self.cache = RunCache(cache_dir or "./cache/agent_runs")

        # Track step results during execution (for caching)
        self.current_run_step_results: Dict[str, List[Dict]] = {}

        self.graph = self._build_graph()
        self.run_checked = False

    @staticmethod
    def _span_name_for_step(step_name: str) -> str:
        return {
            "decide_tool": "tool_choice",
            "lookup_sales_data": "sql_query_exec",
            "analyzing_data": "data_analysis",
            "create_visualization": "gen_visualization",
        }.get(step_name, step_name)

    def check_ollama(self):
        with self.trace_helper.start_span(
            "ollama_check",
            kind="tool",
            input_data={"provider": self.provider, "ollama_url": self.ollama_url},
        ) as span:
            try:
                self.llm.invoke("Hello, how are you?")
                self.trace_helper.set_output(span, {"reachable": True})
                print("Ollama is running locally")
                return True
            except Exception as e:
                self.trace_helper.set_output(span, {"reachable": False, "error": _truncate_trace_text(e)})
                print(e)
                return False

    def check_model(self):
        """Check if the model is running locally (Ollama) or accessible (OpenAI)"""
        with self.trace_helper.start_span(
            "model_access_check",
            kind="tool",
            input_data={"provider": self.provider, "model": self.model},
        ) as span:
            if self.provider == "openai":
                try:
                    self.llm.invoke("Hello")
                    self.trace_helper.set_output(span, {"reachable": True, "provider": self.provider})
                    print("OpenAI API is accessible")
                    return True
                except Exception as e:
                    self.trace_helper.set_output(span, {"reachable": False, "error": _truncate_trace_text(e)})
                    print(f"OpenAI API error: {e}")
                    return False
            else:
                try:
                    base = self.ollama_url.rstrip("/")
                    requests.get(f"{base}/api/version", timeout=3).json()
                    print("Server is running locally")
                    reachable = self.check_ollama()
                    self.trace_helper.set_output(span, {"reachable": reachable, "provider": self.provider})
                    return reachable
                except Exception as e:
                    self.trace_helper.set_output(span, {"reachable": False, "error": _truncate_trace_text(e)})
                    print(e)
                    return False

    def _create_llm(
        self,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        num_beams: int = 1,
        no_repeat_ngram_size: Optional[int] = None,
        callbacks: Optional[list] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
    ):
        """Factory method to create LLM instances with specific parameters.

        Creates a new LLM instance instead of mutating the global self.llm,
        which allows per-step parameter customization.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter (skipped for OpenAI)
            num_beams: Beam search width, 1 = greedy/disabled (skipped for OpenAI)
            no_repeat_ngram_size: Prevent repeating n-grams of this size (skipped for OpenAI)

        Returns:
            ChatOllama or ChatOpenAI instance configured with the given parameters
        """
        _cb = callbacks or []
        resolved_provider = (provider or self.provider).lower()
        resolved_model = model or self.model
        resolved_ollama_url = ollama_url or self.ollama_url

        if resolved_provider == "openai":
            return ChatOpenAI(
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=self.streaming,
                api_key=self.openai_api_key,
                top_p=top_p,
                callbacks=_cb,
            )
        else:
            kwargs = dict(
                model=resolved_model,
                temperature=temperature,
                num_predict=max_tokens,
                streaming=self.streaming,
                base_url=resolved_ollama_url,
                top_p=top_p,
                callbacks=_cb,
            )
            if top_k is not None:
                kwargs["top_k"] = top_k
            if num_beams > 1:
                kwargs["num_beams"] = num_beams
            if no_repeat_ngram_size is not None:
                kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            return ChatOllama(**kwargs)

    def _resolve_step_llm_config(self, config: StepConfig) -> Tuple[str, str, Optional[str]]:
        """Resolve the effective provider/model/base URL for a step.

        Step-level values override the agent defaults when provided.
        """
        provider = (config.provider or self.agent_config.provider or self.provider).lower()
        model = config.model or self.agent_config.model or self.model
        ollama_url = config.ollama_url or self.agent_config.ollama_url or self.ollama_url
        return provider, model, ollama_url

    def _apply_cot_iterations(
        self,
        step_name: str,
        state: State,
        core_fn,
        llm,
        initial_result: Dict,
        cot_n: int,
        trace_helper: Optional[TracingHelper] = None,
    ) -> Dict:
        """Apply up to cot_n iterative CoT refinement steps to a single LLM call.

        Starting from initial_result, repeatedly re-invokes core_fn with a
        CoTRefinementLLM that appends the previous response to every prompt.
        Stops early when the output converges (similarity >= _COT_SIMILARITY_THRESHOLD)
        or after cot_n total iterations (including the initial one).

        Args:
            step_name: Name of the step (for logging and output extraction).
            state: Current agent state (unchanged across iterations).
            core_fn: The core step function, signature (state, llm) -> Dict.
            llm: The base LLM instance (temperature already set by the caller).
            initial_result: Result from the first (non-refinement) call.
            cot_n: Maximum total number of iterations (1 = no refinement).

        Returns:
            The result from the final (or converged) iteration.
        """
        if cot_n <= 1:
            return initial_result

        result = initial_result
        previous_output = _extract_step_output(step_name, result)
        execution_error = result.get("error", "")
        helper = trace_helper or self.trace_helper

        for cot_i in range(1, cot_n):
            print()
            print(f"[{step_name}] CoT iteration {cot_i + 1}/{cot_n}: starting refinement...")
            refinement_llm = CoTRefinementLLM(llm, previous_output, execution_error)
            with helper.start_span(
                "cot_refinement",
                kind="tool",
                attributes={
                    "step_name": step_name,
                    "cot_iteration": cot_i + 1,
                    "cot_total": cot_n,
                },
                input_data={
                    "previous_output": _truncate_trace_text(previous_output),
                    "execution_error": _truncate_trace_text(execution_error),
                },
            ) as span:
                try:
                    new_result = core_fn(state, refinement_llm, trace_helper=helper)
                    helper.set_output(span, _summarize_result_for_trace(new_result))
                except Exception as e:
                    print(f"[{step_name}] CoT iteration {cot_i + 1}/{cot_n} failed: {e}")
                    break

            new_error = new_result.get("error", "")
            if new_error:
                print(
                    f"[{step_name}] CoT iteration {cot_i + 1}/{cot_n}: "
                    f"execution error — {new_error}"
                )
                result = new_result
                previous_output = _extract_step_output(step_name, new_result)
                execution_error = new_error
                continue

            new_output = _extract_step_output(step_name, new_result)
            ratio = difflib.SequenceMatcher(None, previous_output, new_output).ratio()
            print(
                f"[{step_name}] CoT iteration {cot_i + 1}/{cot_n}: "
                f"similarity={ratio:.3f}"
            )

            result = new_result
            execution_error = ""
            helper.set_attributes(
                span,
                {
                    "cot_similarity": float(ratio),
                    "cot_converged": ratio >= _COT_SIMILARITY_THRESHOLD,
                },
            )

            if ratio >= _COT_SIMILARITY_THRESHOLD:
                if cot_i < cot_n - 1:
                    print(
                        f"[{step_name}] CoT early stop: output converged "
                        f"(similarity={ratio:.3f} >= {_COT_SIMILARITY_THRESHOLD})"
                    )
                else:
                    print(
                        f"[{step_name}] Output converged "
                        f"(similarity={ratio:.3f} >= {_COT_SIMILARITY_THRESHOLD})"
                    )
                break

            previous_output = new_output

        return result

    @staticmethod
    def _run_gt_eval(
        step_name: str,
        config: "StepConfig",
        result: Dict,
        state: Dict,
        all_results: Optional[List[Dict]] = None,
    ) -> float:
        """Run ground-truth evaluation for tracking/logging only.

        This NEVER influences selection — it only logs GT scores on the
        already-selected result so performance can be tracked without
        steering the agent.

        Returns
        -------
        float
            Wall-clock time (seconds) spent in LLM judge calls during this
            evaluation, to be accumulated into the per-step LLM call timer.
        """
        if config.gt_eval_fn is None:
            return 0.0

        _gt_t0 = time.perf_counter()

        # Score the selected (best) result
        gt_score = None
        _best_store_snapshot = None
        try:
            gt_score = config.gt_eval_fn(result, state)
            _best_store_snapshot = dict(getattr(config.gt_eval_fn, "_store", {}))
            print(f"[{step_name}] GT tracking score: {gt_score:.3f}")
        except Exception as e:
            print(f"[{step_name}] GT eval error (tracking only): {e}")

        # Score all N candidates for richer tracking
        all_gt_scores = None
        if all_results and len(all_results) > 1:
            all_gt_scores = []
            for r in all_results:
                try:
                    all_gt_scores.append(config.gt_eval_fn(r, state))
                except Exception:
                    all_gt_scores.append(0.0)
            print(f"[{step_name}] All GT scores: {[f'{s:.3f}' for s in all_gt_scores]}")
            # Restore _store to reflect the best result (loop above overwrites it)
            if _best_store_snapshot is not None and hasattr(config.gt_eval_fn, "_store"):
                config.gt_eval_fn._store.clear()
                config.gt_eval_fn._store.update(_best_store_snapshot)

        if gt_score is not None:
            existing = state.get("_gt_scores_per_step") or {}
            existing[step_name] = {
                "gt_score": round(gt_score, 4),
                "all_gt_scores": [round(s, 4) for s in all_gt_scores] if all_gt_scores else None,
            }
            result["_gt_scores_per_step"] = existing

        return time.perf_counter() - _gt_t0

    def _execute_step_with_config(
        self,
        step_name: str,
        state: State,
        core_fn,
        config: StepConfig,
    ) -> Dict:
        """Execute a step with per-step best-of-n, evaluation, and caching.

        This middleware method:
        1. Checks cache if config.use_cache is True
        2. Runs best-of-n sampling if cache miss or force_fresh
        3. Evaluates each of N runs using config.eval_fn
        4. Selects best result using config.selection_fn
        5. Stores all N results for caching

        Args:
            step_name: Name of the step (for logging and caching)
            state: Current agent state
            core_fn: The core step function, signature: (state, llm) -> Dict
            config: StepConfig with parameters for this step

        Returns:
            Updated state dict from the best run
        """
        helper = self.trace_helper
        span_name = self._span_name_for_step(step_name)
        with helper.start_span(
            span_name,
            kind="tool",
            attributes={
                "step_name": step_name,
                "config.enabled": config.enabled,
                "config.cache_mode": getattr(config, "cache_mode", None),
                "config.use_cache": getattr(config, "use_cache", None),
                "config.n": getattr(config, "n", None),
                "config.cot_n": getattr(config, "cot_n", None),
            },
            input_data=_summarize_state_for_trace(state),
        ) as step_span:
            if not config.enabled:
                print(f"[{step_name}] Step disabled, skipping")
                helper.set_output(step_span, {"step_skipped": True})
                return dict(state)

            if config.use_cache and config.cache_mode != "force_fresh":
                with helper.start_span(
                    "cache_lookup",
                    kind="tool",
                    attributes={"step_name": step_name, "cache_mode": config.cache_mode},
                    input_data={"cached_steps": sorted((state.get("cached_step_results") or {}).keys())},
                ) as cache_span:
                    cached_results = state.get("cached_step_results", {})
                    if cached_results and step_name in cached_results:
                        cached = cached_results[step_name]
                        print(f"[{step_name}] Found {len(cached)} cached result(s)")
                        helper.set_output(cache_span, {"cache_hit": True, "cached_result_count": len(cached)})

                        live_csr = state.get("cached_step_results", {})

                        if config.cache_mode == "skip":
                            print(f"[{step_name}] Using cached result (skip mode)")
                            if cached:
                                result = dict(cached[0])
                                result["cached_step_results"] = live_csr
                                self._run_gt_eval(step_name, config, result, state)
                                helper.set_attributes(step_span, {"cache_hit": True, "cache_reused": True})
                                helper.set_output(step_span, _summarize_result_for_trace(result))
                                return result
                            helper.set_output(step_span, {"cache_hit": True, "cached_result_count": 0})
                            return dict(state)

                        if config.eval_fn and len(cached) > 1:
                            scores = []
                            for r in cached:
                                try:
                                    score = config.eval_fn(r, state)
                                except Exception:
                                    score = 0.0
                                scores.append(score)
                            best_idx = config.selection_fn(scores)
                            print(f"[{step_name}] Re-selected cached result {best_idx + 1}/{len(cached)}")
                            result = dict(cached[best_idx])
                            result["cached_step_results"] = live_csr
                            self._run_gt_eval(step_name, config, result, state, all_results=cached)
                            helper.set_attributes(
                                step_span,
                                {
                                    "cache_hit": True,
                                    "cache_reused": True,
                                    "selected_cached_index": best_idx,
                                },
                            )
                            helper.set_output(step_span, _summarize_result_for_trace(result))
                            return result
                        elif cached:
                            result = dict(cached[0])
                            result["cached_step_results"] = live_csr
                            self._run_gt_eval(step_name, config, result, state)
                            helper.set_attributes(step_span, {"cache_hit": True, "cache_reused": True})
                            helper.set_output(step_span, _summarize_result_for_trace(result))
                            return result
                    else:
                        helper.set_output(cache_span, {"cache_hit": False})

            config = self.parameter_provider.get_step_config(step_name, config, state)
            n = config.n
            candidate_params = config.get_candidate_params()
            bon_param = config.bon_param
            helper.set_attributes(
                step_span,
                {
                    "config.n": n,
                    "config.cot_n": config.cot_n,
                    "config.bon_param": bon_param,
                    "config.max_tokens": config.max_tokens,
                },
            )
            step_provider, step_model, step_ollama_url = self._resolve_step_llm_config(config)
            helper.set_attributes(
                step_span,
                {
                    "llm.provider": step_provider,
                    "llm.model": step_model,
                    "llm.ollama_url": step_ollama_url,
                },
            )

            _step_t0 = time.perf_counter()

            # --- Per-step LLM call instrumentation ---
            # _LLMCallAccumulator tracks both time and energy (via per-invoke CodeCarbon
            # trackers) so that measurements cover only the actual LLM inference window.
            # This is correct even when GPU power varies (e.g. local Ollama on A40/L40S),
            # since no pro-rating assumption is made.
            _llm_invoke_time = 0.0
            _cc_step_dir = None
            if getattr(self, '_enable_codecarbon_steps', False) and _CODECARBON_AVAILABLE:
                _cc_step_dir = os.path.join(
                    self._codecarbon_step_base_dir, f"step_{step_name}"
                )
                os.makedirs(_cc_step_dir, exist_ok=True)
            _llm_acc = _LLMCallAccumulator(
                cc_enabled=getattr(self, '_enable_codecarbon_steps', False),
                cc_output_dir=_cc_step_dir,
            )

            if n == 1:
                temp, top_p, top_k = candidate_params[0]
                llm = self._create_llm(
                    temperature=temp,
                    max_tokens=config.max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    callbacks=[_llm_acc],
                    provider=step_provider,
                    model=step_model,
                    ollama_url=step_ollama_url,
                )
                try:
                    with helper.start_span(
                        "step_candidate",
                        kind="tool",
                        attributes={
                            "step_name": step_name,
                        "candidate_index": 0,
                        "temperature": temp,
                        "top_p": top_p,
                        "top_k": top_k,
                        "llm.provider": step_provider,
                        "llm.model": step_model,
                    },
                    ) as candidate_span:
                        if config.cot_n > 1:
                            print(f"[{step_name}] CoT iteration 1/{config.cot_n}: starting initial run...")
                        result = core_fn(state, llm, trace_helper=helper)
                        result["_temperature"] = temp
                        result["_top_p"] = top_p
                        result["_top_k"] = top_k
                        result["_run_idx"] = 0
                        result = self._apply_cot_iterations(
                            step_name,
                            state,
                            core_fn,
                            llm,
                            result,
                            config.cot_n,
                            trace_helper=helper,
                        )
                        result["_temperature"] = temp
                        result["_top_p"] = top_p
                        result["_top_k"] = top_k
                        result["_run_idx"] = 0
                        helper.set_output(candidate_span, _summarize_result_for_trace(result))
                except Exception as e:
                    print(f"[{step_name}] Error: {e}")
                    result = dict(state)
                    result["error"] = str(e)

                self.current_run_step_results[step_name] = [result]

                # Apply column standardization for lookup_sales_data even when n=1
                # (the n>1 path has its own block at line ~752 after collecting all candidates)
                if step_name == "lookup_sales_data" and getattr(config, 'gt_columns', None):
                    try:
                        from Agent.utils import standardize_candidate_columns
                        standardize_llm = self._create_llm(
                            temperature=0.0,
                            max_tokens=1000,
                            callbacks=[_llm_acc],
                            provider=step_provider,
                            model=step_model,
                            ollama_url=step_ollama_url,
                        )
                        std_results = standardize_candidate_columns(
                            [result], self.schema, standardize_llm,
                            gt_columns=getattr(config, 'gt_columns', None),
                        )
                        result = std_results[0]
                        self.current_run_step_results[step_name] = [result]
                        helper.set_attributes(step_span, {"standardized_candidate_columns": True})
                    except Exception as e:
                        print(f"[{step_name}] Column standardization warning: {e}")

                _llm_invoke_time += self._run_gt_eval(step_name, config, result, state)
                eval_score = None
                if config.eval_fn:
                    try:
                        _t = time.perf_counter()
                        eval_score = config.eval_fn(result, state)
                        _llm_invoke_time += time.perf_counter() - _t
                    except Exception:
                        pass
                elif config.batch_eval_fn:
                    try:
                        _t = time.perf_counter()
                        batch_scores = config.batch_eval_fn([result], state)
                        _llm_invoke_time += time.perf_counter() - _t
                        eval_score = batch_scores[0] if batch_scores else None
                    except Exception:
                        pass
                # Add callback-accumulated LLM invoke time (core_fn + CoT + standardize)
                _llm_invoke_time += _llm_acc.total_time
                _step_elapsed = time.perf_counter() - _step_t0
                existing_timings = state.get("_step_timings_sec") or {}
                existing_timings[step_name] = round(_step_elapsed, 3)
                result["_step_timings_sec"] = existing_timings
                if eval_score is not None:
                    existing_eval = state.get("_step_eval_scores") or {}
                    existing_eval[step_name] = {
                        "scores": [round(eval_score, 4)],
                        "best_idx": 0,
                        "best_score": round(eval_score, 4),
                    }
                    result["_step_eval_scores"] = existing_eval
                # --- Store per-step LLM call metrics ---
                _existing_llm_time = state.get("_step_llm_timings_sec") or {}
                _existing_llm_time[step_name] = round(_llm_invoke_time, 3)
                result["_step_llm_timings_sec"] = _existing_llm_time
                if _llm_acc._cc_enabled and _llm_acc.total_energy["energy_consumed_kwh"] > 0:
                    _existing_llm_energy = state.get("_step_llm_energy") or {}
                    _existing_llm_energy[step_name] = dict(_llm_acc.total_energy)
                    result["_step_llm_energy"] = _existing_llm_energy
                helper.set_output(step_span, _summarize_result_for_trace(result))
                return result

            results = []
            scores = []

            _param_idx = {"temperature": 0, "top_p": 1, "top_k": 2}[bon_param]
            varying_vals = [p[_param_idx] for p in candidate_params]
            print(f"[{step_name}] Running best-of-{n} varying {bon_param}: {varying_vals}")
            helper.set_attributes(step_span, {"candidate_count": n, "varying_values": varying_vals})

            for i, (temp, top_p, top_k) in enumerate(candidate_params):
                if i > 0:
                    print()
                    print()
                llm = self._create_llm(
                    temperature=temp,
                    max_tokens=config.max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    callbacks=[_llm_acc],
                    provider=step_provider,
                    model=step_model,
                    ollama_url=step_ollama_url,
                )
                varying_val = varying_vals[i]

                try:
                    with helper.start_span(
                        "step_candidate",
                        kind="tool",
                        attributes={
                            "step_name": step_name,
                            "candidate_index": i,
                            "temperature": temp,
                            "top_p": top_p,
                            "top_k": top_k,
                            bon_param: varying_val,
                            "llm.provider": step_provider,
                            "llm.model": step_model,
                        },
                    ) as candidate_span:
                        if config.cot_n > 1:
                            print(f"[{step_name}] CoT iteration 1/{config.cot_n}: starting initial run...")
                        result = core_fn(state, llm, trace_helper=helper)
                        result["_temperature"] = temp
                        result["_top_p"] = top_p
                        result["_top_k"] = top_k
                        result["_bon_param"] = bon_param
                        result["_run_idx"] = i
                        result = self._apply_cot_iterations(
                            step_name,
                            state,
                            core_fn,
                            llm,
                            result,
                            config.cot_n,
                            trace_helper=helper,
                        )
                        result["_temperature"] = temp
                        result["_top_p"] = top_p
                        result["_top_k"] = top_k
                        result["_bon_param"] = bon_param
                        result["_run_idx"] = i

                        if config.eval_fn:
                            try:
                                _t = time.perf_counter()
                                score = config.eval_fn(result, state)
                                _llm_invoke_time += time.perf_counter() - _t
                            except Exception as eval_err:
                                print(f"  Run {i + 1}/{n}: eval error: {eval_err}")
                                score = 0.0
                            print(f"  Run {i + 1}/{n} ({bon_param}={varying_val}): score={score:.3f}")
                        elif config.batch_eval_fn:
                            score = 0.0
                            print(f"  Run {i + 1}/{n} ({bon_param}={varying_val}): score=pending (batch eval)")
                        else:
                            score = 0.0
                            print(f"  Run {i + 1}/{n} ({bon_param}={varying_val}): done (no evaluator set)")

                        helper.set_attributes(candidate_span, {"candidate_score": float(score)})
                        helper.set_output(candidate_span, _summarize_result_for_trace(result))
                        results.append(result)
                        scores.append(score)
                except Exception as e:
                    print(f"  Run {i + 1}/{n} failed: {e}")
                    error_result = dict(state)
                    error_result["error"] = str(e)
                    error_result["_temperature"] = temp
                    error_result["_top_p"] = top_p
                    error_result["_top_k"] = top_k
                    error_result["_bon_param"] = bon_param
                    error_result["_run_idx"] = i
                    results.append(error_result)
                    scores.append(-float("inf"))

            self.current_run_step_results[step_name] = results

            if step_name == "lookup_sales_data" and (len(results) > 1 or getattr(config, 'gt_columns', None)):
                try:
                    from Agent.utils import standardize_candidate_columns
                    standardize_llm = self._create_llm(
                        temperature=0.0,
                        max_tokens=1000,
                        callbacks=[_llm_acc],
                        provider=step_provider,
                        model=step_model,
                        ollama_url=step_ollama_url,
                    )
                    results = standardize_candidate_columns(
                        results, self.schema, standardize_llm,
                        gt_columns=getattr(config, 'gt_columns', None),
                    )
                    self.current_run_step_results[step_name] = results
                    helper.set_attributes(step_span, {"standardized_candidate_columns": True})
                except Exception as e:
                    print(f"[{step_name}] Column standardization warning: {e}")

            if config.batch_eval_fn:
                try:
                    _t = time.perf_counter()
                    scores = config.batch_eval_fn(results, state)
                    _llm_invoke_time += time.perf_counter() - _t
                    print(f"[{step_name}] Batch eval scores: {[f'{s:.3f}' for s in scores]}")
                except Exception as e:
                    print(f"[{step_name}] Batch eval error: {e}")

            if not scores or all(s == -float("inf") for s in scores):
                best_result = results[0] if results else dict(state)
                best_idx = 0 if results else None
            else:
                best_idx = config.selection_fn(scores)
                best_result = results[best_idx]
                best_result["_best_idx"] = best_idx
                best_result["_all_scores"] = scores
                print(f"[{step_name}] Selected run {best_idx + 1}/{n} (score={scores[best_idx]:.3f})")

            _llm_invoke_time += self._run_gt_eval(step_name, config, best_result, state, all_results=results)
            # Add callback-accumulated LLM invoke time (core_fn + CoT + standardize across all candidates)
            _llm_invoke_time += _llm_acc.total_time
            _step_elapsed = time.perf_counter() - _step_t0
            existing_timings = state.get("_step_timings_sec") or {}
            existing_timings[step_name] = round(_step_elapsed, 3)
            best_result["_step_timings_sec"] = existing_timings
            existing_eval = state.get("_step_eval_scores") or {}
            existing_eval[step_name] = {
                "scores": [round(s, 4) for s in scores],
                "best_idx": best_idx,
                "best_score": round(scores[best_idx], 4) if best_idx is not None and scores else None,
            }
            best_result["_step_eval_scores"] = existing_eval
            # --- Store per-step LLM call metrics ---
            _existing_llm_time = state.get("_step_llm_timings_sec") or {}
            _existing_llm_time[step_name] = round(_llm_invoke_time, 3)
            best_result["_step_llm_timings_sec"] = _existing_llm_time
            if _llm_acc._cc_enabled and _llm_acc.total_energy["energy_consumed_kwh"] > 0:
                _existing_llm_energy = state.get("_step_llm_energy") or {}
                _existing_llm_energy[step_name] = dict(_llm_acc.total_energy)
                best_result["_step_llm_energy"] = _existing_llm_energy
            helper.set_attributes(
                step_span,
                {
                    "selected_candidate_index": best_idx,
                    "all_scores": [float(score) for score in scores[:_TRACE_LIST_LIMIT]],
                },
            )
            helper.set_output(step_span, _summarize_result_for_trace(best_result))
            return best_result

    def _maybe_save_run_results(
        self,
        run_id: str,
        prompt: str,
        result: Dict,
        save_results: bool
    ) -> None:
        """Save run results to cache if save_results is True.

        Args:
            run_id: Unique identifier for this run
            prompt: User prompt that initiated this run
            result: Final result from the agent
            save_results: Whether to actually save
        """
        if not save_results:
            return

        with self.trace_helper.start_span(
            "cache_save_run",
            kind="tool",
            input_data={"run_id": run_id, "prompt": _truncate_trace_text(prompt)},
        ) as span:
            try:
                self.cache.save_run(
                    run_id=run_id,
                    prompt=prompt,
                    agent_config=self.agent_config.to_dict(),
                    step_results=self.current_run_step_results,
                    final_result=result,
                    metadata={}
                )
                self.trace_helper.set_output(
                    span,
                    {
                        "run_id": run_id,
                        "saved": True,
                        "step_result_count": len(self.current_run_step_results),
                    },
                )
                print(f"[Agent] Run saved with ID: {run_id}")
            except Exception as e:
                self.trace_helper.set_output(span, {"run_id": run_id, "saved": False, "error": _truncate_trace_text(e)})
                print(f"[Agent] Warning: Failed to save run to cache: {e}")

    def _build_graph(self):
        """Construct and compile the LangGraph for the agent run loop.

        Uses the middleware pattern to support per-step configuration including
        best-of-n sampling, custom evaluation, and caching. Each node wraps
        a *_core function with _execute_step_with_config().
        """
        graph = StateGraph(State)

        # Factory to create configured node functions
        def make_configured_node(step_name: str, core_fn):
            """Create a node function that uses per-step configuration."""
            def node_fn(state: State) -> Dict:
                default_config = self.agent_config.get_step_config(step_name)
                return self._execute_step_with_config(step_name, state, core_fn, default_config)
            return node_fn

        # Bind schema into lookup_sales_data_core via partial so the middleware
        # signature core_fn(state, llm) is preserved.
        lookup_core_with_schema = partial(lookup_sales_data_core, schema=self.schema)

        # Add nodes with configuration wrappers
        graph.add_node("decide_tool", make_configured_node("decide_tool", decide_tool_core))
        graph.add_node("lookup_sales_data", make_configured_node("lookup_sales_data", lookup_core_with_schema))
        graph.add_node("analyzing_data", make_configured_node("analyzing_data", analyzing_data_core))
        graph.add_node("create_visualization", make_configured_node("create_visualization", create_visualization_core))

        graph.set_entry_point("decide_tool")

        # Routing logic (unchanged)
        graph.add_conditional_edges(
            "decide_tool",
            route_to_tool,
            {
                "lookup_sales_data": "lookup_sales_data",
                "analyzing_data": "analyzing_data",
                "create_visualization": "create_visualization",
                "end": END,
            },
        )

        graph.add_edge("lookup_sales_data", "decide_tool")
        graph.add_edge("analyzing_data", "decide_tool")
        graph.add_edge("create_visualization", "decide_tool")

        return graph.compile()
    
    def draw_graph(self) -> str:
        """Return an ASCII rendering of the compiled graph if available."""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # Fallback if mermaid is not available
            print(self.graph.get_graph().print_ascii())

    def run_core(
        self,
        prompt: str,
        *,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        # New: caching parameters
        run_id: Optional[str] = None,
        cached_step_results: Optional[Dict] = None,
        save_results: bool = False,
    ) -> Dict:
        """Execute the agent for a single prompt.

        Args:
            prompt: Natural-language request or question.
            visualization_goal: Optional explicit goal for charts; defaults to the prompt.
            lookup_only: Only run data lookup step.
            no_vis: Skip visualization step.
            run_id: Unique ID for this run (for caching).
            cached_step_results: Pre-loaded cached results from similar past runs.
            save_results: Whether to save this run's results to cache.

        Returns:
            The final state dictionary produced by the compiled graph execution.
        """
        import uuid

        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        # Reset step results tracker
        self.current_run_step_results = {}
        _run_t0 = time.perf_counter()

        # Initialize state with caching info
        state = {
            "prompt": prompt,
            "run_id": run_id,
            "cached_step_results": cached_step_results or {},
        }
        if visualization_goal:
            state["visualization_goal"] = visualization_goal
        run_span_name = "AgentRun_LookupOnly" if lookup_only else "AgentRun_NoVis" if no_vis else "AgentRun"
        with self.trace_helper.start_span(
            run_span_name,
            kind="agent",
            attributes={
                "run_id": run_id,
                "provider": self.provider,
                "model": self.model,
                "lookup_only": lookup_only,
                "no_vis": no_vis,
                "tracing_enabled": self.tracing_enabled,
                "cached_step_count": len(cached_step_results or {}),
            },
            input_data=_summarize_state_for_trace(state),
        ) as run_span:
            if not self.run_checked:
                print("Checking the model can run locally")
                self.run_checked = self.check_model()

            if not self.run_checked:
                error_msg = "Model is not accessible. " + (
                    "Remember to run 'ollama serve' for Ollama models." if self.provider == "ollama"
                    else "Check your OpenAI API key and internet connection."
                )
                print(error_msg)
                result = {**state, "error": error_msg}
                self.trace_helper.set_output(run_span, _summarize_result_for_trace(result))
                return result

            if lookup_only:
                print("[Agent] Running only lookup_sales_data")
                try:
                    lookup_cfg = self.agent_config.get_step_config("lookup_sales_data")
                    lookup_core = partial(lookup_sales_data_core, schema=self.schema)
                    result = self._execute_step_with_config("lookup_sales_data", state, lookup_core, lookup_cfg)
                    self._maybe_save_run_results(run_id, prompt, result, save_results)
                    result["run_id"] = run_id
                    result["_total_run_time_sec"] = round(time.perf_counter() - _run_t0, 3)
                    self.trace_helper.set_output(run_span, _summarize_result_for_trace(result))
                    return result
                except Exception as _e:
                    result = {**state, "error": f"Lookup failed: {str(_e)}"}
                    self.trace_helper.set_output(run_span, _summarize_result_for_trace(result))
                    return result

            if no_vis:
                print("[Agent] Running agent without visualization")
                try:
                    lookup_cfg = self.agent_config.get_step_config("lookup_sales_data")
                    analyzing_cfg = self.agent_config.get_step_config("analyzing_data")
                    lookup_core = partial(lookup_sales_data_core, schema=self.schema)
                    print("\n\nTool selected: lookup_sales_data")
                    state = self._execute_step_with_config("lookup_sales_data", state, lookup_core, lookup_cfg)
                    print("\n\nTool selected: analyzing_data")
                    result = self._execute_step_with_config("analyzing_data", state, analyzing_data_core, analyzing_cfg)
                    print(f"\nAgent response: {result.get('answer', [None])[0]}")
                    self._maybe_save_run_results(run_id, prompt, result, save_results)
                    result["run_id"] = run_id
                    result["_total_run_time_sec"] = round(time.perf_counter() - _run_t0, 3)
                    self.trace_helper.set_output(run_span, _summarize_result_for_trace(result))
                    return result
                except Exception as _e:
                    print(f"Lookup failed: {str(_e)}")
                    result = {**state, "error": f"Lookup failed: {str(_e)}"}
                    self.trace_helper.set_output(run_span, _summarize_result_for_trace(result))
                    return result

            print("Running the graph...")
            result = self.graph.invoke(state)
            print(f"\nAgent response: {result.get('answer', [])}")
            print("[LangGraph] LangGraph execution completed")
            self._maybe_save_run_results(run_id, prompt, result, save_results)
            result["run_id"] = run_id
            result["_total_run_time_sec"] = round(time.perf_counter() - _run_t0, 3)
            self.trace_helper.set_output(run_span, _summarize_result_for_trace(result))
            return result
    
    def _run_with_evaluation(
        self,
        *,
        prompt: str,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        best_of_n: int = 1,
        temp: Optional[float] = None,
        temp_max: Optional[float] = None,
        csv_eval_fn: Optional[callable] = None,
        text_eval_fn: Optional[callable] = None,
        vis_eval_fn: Optional[callable] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """Core evaluation logic extracted from run() for CodeCarbon wrapping."""
        
        if best_of_n > 1 and temp is not None and temp_max is not None:
            temps = np.linspace(temp, temp_max, best_of_n).tolist()
        else:
            temps = [temp if temp is not None else self.llm.temperature] * best_of_n
        
        print(f"[Agent] Running best-of-{best_of_n} with temperatures: {temps}")
        
        all_results = []
        all_scores = []
        
        for i in range(best_of_n):
            original_temp = self.llm.temperature
            self.llm.temperature = temps[i]
            
            try:
                result = self.run_core(
                    prompt,
                    visualization_goal=visualization_goal,
                    lookup_only=lookup_only,
                    no_vis=no_vis
                )

                # Save CSV
                csv_path = None
                if result.get("data"):
                    csv_path = os.path.join(save_dir, f"run_data.csv")
                    result_rows = text_to_csv(result['data'])
                    save_csv(result_rows, csv_path)
                
                # Extract analysis text
                analysis_text = result.get("answer", [None])[0] if result.get("answer") else None
                
                # Evaluate
                score = 0.0
                csv_score = None
                text_score = None
                
                if csv_eval_fn:
                    csv_score = csv_eval_fn(csv_path)
                    score += csv_score
                    result["csv_score"] = csv_score
                
                if text_eval_fn:
                    text_score = text_eval_fn(analysis_text)
                    score += text_score
                    result["text_score"] = text_score

                # Visualization evaluation
                if vis_eval_fn and not no_vis and not lookup_only:
                    chart_config = result.get("chart_config")
                    # Chart code is the last answer entry (after analysis text)
                    answers = result.get("answer", [])
                    chart_code = answers[-1] if len(answers) > 1 else None

                    if chart_config and chart_code:
                        vis_score = vis_eval_fn(chart_config, chart_code)
                        score += vis_score
                        result["vis_score"] = vis_score

                result["temperature"]= temps[i]

                all_results.append(result)
                all_scores.append(score)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
        self.llm.temperature = original_temp
        print(all_scores)
        if not all_scores:
            return {}, 0.0
        
        best_idx = int(np.argmax(all_scores))
        best_result = all_results[best_idx]
        
        results_path = os.path.join(save_dir, "all_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        score_variance = (max(all_scores) - min(all_scores))/max(all_scores) if max(all_scores) != 0 else 0.0
        return best_result, score_variance
            
    def run(
        self,
        prompt: str,
        *,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        best_of_n: int = 1,
        temp: Optional[float] = None,
        temp_max: Optional[float] = None,
        csv_eval_fn: Optional[callable] = None,
        text_eval_fn: Optional[callable] = None,
        vis_eval_fn: Optional[callable] = None,
        save_dir: Optional[str] = None,
        enable_codecarbon: bool = False,
        # New: caching parameters
        run_id: Optional[str] = None,
        reuse_from: Optional[str] = None,
        step_overrides: Optional[Dict[str, Dict]] = None,
        save_results: bool = False,
    ) -> Dict:
        """Run the agent with optional caching and per-step configuration.

        Args:
            prompt: User query/question.
            visualization_goal: Optional explicit visualization goal.
            lookup_only: Only run data lookup step.
            no_vis: Skip visualization step.
            best_of_n: (Deprecated) Number of agent-level runs for old best-of-n.
                       Use step-level configuration via AgentConfig instead.
            temp, temp_max: (Deprecated) Temperature range for old best-of-n.
            csv_eval_fn, text_eval_fn, vis_eval_fn: (Deprecated) Evaluation functions
                       for old agent-level best-of-n.
            save_dir: Directory for saving results (old API).
            enable_codecarbon: Enable carbon emissions tracking.
            reuse_from: Run ID to load cached results from (new caching API).
            step_overrides: Dict mapping step_name -> config overrides for this run.
                           Example: {"analyzing_data": {"n": 10, "temp_max": 0.9}}
            save_results: Whether to save this run's results to cache.

        Returns:
            Result dict with 'answer', 'data', 'chart_config', etc.
            Includes 'run_id' if save_results=True.
        """

        if save_dir is None:
            save_dir = tempfile.mkdtemp(prefix="agent_runs_")
        os.makedirs(save_dir, exist_ok=True)

        # Apply step overrides if provided
        original_config = None
        if step_overrides:
            from copy import deepcopy
            original_config = self.agent_config
            self.agent_config = deepcopy(self.agent_config)
            for step_name, overrides in step_overrides.items():
                step_config = self.agent_config.get_step_config(step_name)
                for key, value in overrides.items():
                    if hasattr(step_config, key):
                        setattr(step_config, key, value)

        # Find/load cached results
        cached_step_results = {}
        if reuse_from:
            # Load from specific run
            cached_step_results = self.cache.load_all_step_results(reuse_from)
            if cached_step_results:
                print(f"[Agent] Loaded cached results from run: {reuse_from}")
            else:
                print(f"[Agent] Warning: No cached results found for run: {reuse_from}")
        elif save_results:
            # Auto-find similar runs
            similar_runs = self.cache.find_similar_runs(prompt, top_k=3)
            if similar_runs:
                print(f"[Agent] Found {len(similar_runs)} similar run(s): {similar_runs}")
                cached_step_results = self.cache.load_all_step_results(similar_runs[0])

        # Use new API unless the caller is explicitly using the deprecated agent-level
        # best-of-n or old-style eval functions (csv_eval_fn / text_eval_fn / vis_eval_fn).
        use_new_api = (
            best_of_n == 1
            and csv_eval_fn is None
            and text_eval_fn is None
            and vis_eval_fn is None
        )
        if use_new_api:
            try:
                tracker = None
                if enable_codecarbon and _CODECARBON_AVAILABLE:
                    codecarbon_dir = os.path.join(save_dir, "codecarbon")
                    os.makedirs(codecarbon_dir, exist_ok=True)
                    try:
                        tracker = EmissionsTracker(  # type: ignore[call-arg]
                            project_name="SalesDataAgent",
                            output_dir=codecarbon_dir,
                            save_to_file=True,
                            measure_power_secs=1,
                            log_level="error",
                            allow_multiple_runs=True,
                        )
                        tracker.start()
                    except Exception as e:
                        print(f"CodeCarbon tracking failed to start: {e}, continuing without it")
                        tracker = None
                # Enable per-step LLM call tracking
                self._enable_codecarbon_steps = enable_codecarbon
                self._codecarbon_step_base_dir = (
                    os.path.join(save_dir, "codecarbon_steps") if enable_codecarbon else None
                )
                if self._codecarbon_step_base_dir:
                    os.makedirs(self._codecarbon_step_base_dir, exist_ok=True)
                try:
                    result = self.run_core(
                        prompt,
                        visualization_goal=visualization_goal,
                        lookup_only=lookup_only,
                        no_vis=no_vis,
                        run_id=run_id,
                        cached_step_results=cached_step_results,
                        save_results=save_results,
                    )
                finally:
                    self._enable_codecarbon_steps = False
                    self._codecarbon_step_base_dir = None
                    if tracker is not None:
                        try:
                            tracker.stop()
                            if hasattr(tracker, "final_emissions_data") and tracker.final_emissions_data is not None:
                                ed = tracker.final_emissions_data
                                result["_energy"] = {
                                    "energy_consumed_kwh": ed.energy_consumed,
                                    "cpu_energy_kwh": ed.cpu_energy,
                                    "gpu_energy_kwh": ed.gpu_energy,
                                    "ram_energy_kwh": ed.ram_energy,
                                    "emissions_kg_co2": ed.emissions,
                                    "cpu_power_w": ed.cpu_power,
                                    "gpu_power_w": ed.gpu_power,
                                    "duration_sec": ed.duration,
                                }
                        except Exception as e:
                            print(f"CodeCarbon tracking failed to stop: {e}")
                return result
            finally:
                # Restore original config if we modified it
                if original_config is not None:
                    self.agent_config = original_config

        # Restore original config before falling through to old API
        if original_config is not None:
            self.agent_config = original_config

        # Wrap execution with CodeCarbon if requested and available
        if enable_codecarbon and _CODECARBON_AVAILABLE:
            codecarbon_dir = os.path.join(save_dir, "codecarbon")
            os.makedirs(codecarbon_dir, exist_ok=True)
            try:
                with EmissionsTracker(  # type: ignore[call-arg]
                    project_name="SalesDataAgent",
                    output_dir=codecarbon_dir,
                    save_to_file=True,
                    measure_power_secs=1,
                    log_level="error",
                    allow_multiple_runs=False,
                ):
                    return self._run_with_evaluation(
                        prompt=prompt,
                        visualization_goal=visualization_goal,
                        lookup_only=lookup_only,
                        no_vis=no_vis,
                        best_of_n=best_of_n,
                        temp=temp,
                        temp_max=temp_max,
                        csv_eval_fn=csv_eval_fn,
                        text_eval_fn=text_eval_fn,
                        vis_eval_fn=vis_eval_fn,
                        save_dir=save_dir,
                    )
            except Exception as e:
                print(f"CodeCarbon tracking failed: {e}, continuing without it")
                # Fall through to run without CodeCarbon

        return self._run_with_evaluation(
            prompt=prompt,
            visualization_goal=visualization_goal,
            lookup_only=lookup_only,
            no_vis=no_vis,
            best_of_n=best_of_n,
            temp=temp,
            temp_max=temp_max,
            csv_eval_fn=csv_eval_fn,
            text_eval_fn=text_eval_fn,
            vis_eval_fn=vis_eval_fn,
            save_dir=save_dir,
        )

__all__ = ["SalesDataAgent", "State"]
