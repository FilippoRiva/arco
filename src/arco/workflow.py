"""Sales Data Agent using LangGraph, DuckDB, and Ollama (LLaMA).

This module exposes a class `SalesDataAgent` that orchestrates:
- DuckDB SQL over a local parquet file
- LLM-driven tool routing (lookup → analyze → visualize)
- Chart configuration extraction and chart code generation

Usage example:
    from arco.workflow import SalesDataWorkflow

    workflow = SalesDataWorkflow()
    result = workflow.run("Show me the sales in Nov 2021")
"""
from typing import Any
from typing import Generator
from arco.llm_tools import LLMCallAccumulator
import json
import logging
import os
import re
import time

import requests

# codecarbon
from codecarbon import EmissionsTracker

logging.getLogger("codecarbon").setLevel(logging.ERROR)  # Hide codecarbon warnings

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from arco import llm_tools, tracing
from arco.agents import Analyzer, Orchestrator, Retriever, Visualizer
from arco.core import ArcoConfig, State, AgentType
from arco.data import RunCache
from arco.global_vars import CODECARBON_AVAILABLE
from arco.tracing import (truncate_trace_text, _summarize_for_trace, TracingHelper)


def serialize_for_json(value):
    """Convert results into a JSON-safe structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]

    if value.__class__.__name__ == "DataFrame" and hasattr(value, "to_dict"):
        # noinspection PyBroadException
        try:
            return {
                "__dataframe__": True,
                "records": serialize_for_json(value.to_dict(orient="records")),
            }
        except Exception:
            return str(value)

    if value.__class__.__name__ in ("Timestamp", "NaTType"):
        return str(value)

    if hasattr(value, "item"):  # numpy scalar (int64, float64, etc.)
        return value.item()

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _slugify_prompt(prompt, max_len=48):
    """Return a filesystem-safe prompt slug."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", (prompt or "").strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        return "run"
    return normalized[:max_len].rstrip("_") or "run"


class SalesDataWorkflow:
    """End-to-end agentic workflow to query, analyze, and visualize sales data.

    The agent builds a LangGraph with tool-selection, data lookup (DuckDB over
    parquet), LLM-based analysis, and visualization code generation. Use `run()`
    to execute a single prompt through the flow.
    """

    def __init__(
            self,
            *,
            config: ArcoConfig
    ) -> None:
        """Initialize the agent and compile the graph. If the config parameter is set, all other parameters are ignored

        Args:
            model: Model name (OpenAI model like "gpt-4o-mini" or Ollama model like "llama3.2:3b").
            provider: LLM provider to use ("ollama" or "openai"). Default is "ollama".
            ollama_url: Optional override for Ollama base URL; defaults to OLLAMA_HOST or http://localhost:11434.
            config: Optional GlobalConfig for complete control on execution
        """
        self.config = config
        self.model_is_reachable = False

        # Tracing
        self.tracer = None
        if config.enable_tracing:
            # Environment variables similar to utils_0.py
            tracing.init_tracing(config.phoenix_endpoint)
            self.trace_helper = tracing.get_tracer(project_name=config.phoenix_project_name)
        else :
            self.trace_helper = TracingHelper()

        # Caching
        self.cache = RunCache(config.save_dir) if config.save_dir else RunCache()

        # Initialize graph
        self.graph: CompiledStateGraph = self._build_graph()

    def _strict_graph(self) -> CompiledStateGraph:
        graph = StateGraph(State)

        init_args = {
            "trace_helper": self.trace_helper
        }

        # Add nodes
        graph.add_node(
            AgentType.RETRIEVER.value,
            Retriever(schema=self.config.schema, **init_args).get_node())
        graph.add_node(
            AgentType.ANALYZER.value,
            Analyzer(**init_args).get_node())
        graph.add_node(
            AgentType.VISUALIZER.value,
            Visualizer(**init_args).get_node())

        graph.set_entry_point(AgentType.RETRIEVER.value)

        graph.add_edge(
            AgentType.RETRIEVER.value,
            AgentType.ANALYZER.value)
        graph.add_edge(
            AgentType.ANALYZER.value,
            AgentType.VISUALIZER.value)
        graph.add_edge(
            AgentType.VISUALIZER.value,
            END)

        return graph.compile()

    def _orchestration_graph(self) -> CompiledStateGraph:
        graph = StateGraph(State)

        init_args = {
            "trace_helper": self.trace_helper
        }

        # Add nodes
        graph.add_node(
            AgentType.ORCHESTRATOR.value,
            Orchestrator(**init_args).get_node())
        graph.add_node(
            AgentType.RETRIEVER.value,
            Retriever(schema=self.config.schema, **init_args).get_node())
        graph.add_node(
            AgentType.ANALYZER.value,
            Analyzer(**init_args).get_node())
        graph.add_node(
            AgentType.VISUALIZER.value,
            Visualizer(**init_args).get_node())

        graph.set_entry_point(AgentType.ORCHESTRATOR.value)

        def route_to_agent(state: State) -> str:
            answer = state.get_last_answer(AgentType.ORCHESTRATOR)
            valid_choices = [
                AgentType.RETRIEVER.value,
                AgentType.ANALYZER.value,
                AgentType.VISUALIZER.value
            ]
            if answer and answer.agent_choice and answer.agent_choice in valid_choices:
                return answer.agent_choice
            return "end"

        # Routing logic
        graph.add_conditional_edges(
            AgentType.ORCHESTRATOR.value,
            route_to_agent,
            {
                AgentType.RETRIEVER.value: AgentType.RETRIEVER.value,
                AgentType.ANALYZER.value: AgentType.ANALYZER.value,
                AgentType.VISUALIZER.value: AgentType.VISUALIZER.value,
                "end": END,
            },
        )

        # Edges returning to orchestrator
        graph.add_edge(AgentType.RETRIEVER.value, AgentType.ORCHESTRATOR.value)
        graph.add_edge(AgentType.ANALYZER.value, AgentType.ORCHESTRATOR.value)
        graph.add_edge(AgentType.VISUALIZER.value, AgentType.ORCHESTRATOR.value)

        return graph.compile()

    def _build_graph(self) -> CompiledStateGraph:
        """Construct and compile the LangGraph for the agent run loop."""
        if self.config.orchestration_enabled:
            return self._orchestration_graph()
        return self._strict_graph()

    def draw_graph(self):
        """Return an ASCII rendering of the compiled graph if available."""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except ImportError:
            # Fallback if mermaid is not available
            self.graph.get_graph().print_ascii()

    def run(self) -> Generator[dict[str, Any]]:
        yield {"event": "started", "run_id": self.config.run_id, "config": self.config}

        # Cache load
        cached_results = {}
        if self.config.use_cache and self.config.cache_mode in ["read", "r", "read_write", "rw"]:
            yield {"event": "cache", "value": "start"}
            # Auto-find similar runs
            similar_runs = self.cache.find_similar_runs(self.config.prompt, top_k=3)
            if similar_runs:
                cached_results = self.cache.load_all_step_results(similar_runs[0])
                yield {"event": "cache", "value": "hit"}
            else:
                yield {"event": "cache", "value": "miss"}

        # Initialize state with loaded cache results
        input_state: State = State(
            prompt=self.config.prompt,
            run_id=self.config.run_id,
            cached_results=cached_results,
            visualization_goal=self.config.visualization_goal,
            agent_configs=self.config.agent_configs
        )

        # codecarbon setup
        tracker = None
        save_dir = self.config.save_dir if self.config.save_dir else "./output"
        if self.config.enable_codecarbon and CODECARBON_AVAILABLE:
            # Global Emission Tracking
            codecarbon_dir = os.path.join(save_dir, "codecarbon")
            os.makedirs(codecarbon_dir, exist_ok=True)
            tracker = EmissionsTracker(
                project_name="SalesDataAgent",
                output_dir=codecarbon_dir,
                save_to_file=True,
                measure_power_secs=1,
                log_level="error",
                allow_multiple_runs=True,
                experiment_id=self.config.run_id
            )
            tracker.start()
            codecarbon_base_dir = os.path.join(save_dir, "codecarbon")
            os.makedirs(codecarbon_base_dir, exist_ok=True)
            # LLM Emission Tracking
            LLMCallAccumulator.enable(codecarbon_base_dir)

        # Check Model Reachability
        if not self.model_is_reachable:
            self.model_is_reachable = self.check_model()
            if not self.model_is_reachable:
                yield {"event": "error", "message": "Model is not reachable. Please set your OPENAI_API_KEY environment variable if using openai models or properly start the ollama server."}
                return

        # Start Inference and Generator Loop
        _run_t0 = time.perf_counter()

        with (self.trace_helper.start_span(
                "AgentRun",
                kind="agent",
                attributes={
                    "run_id": self.config.run_id,
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "tracing_enabled": self.config.tracing['enabled'],
                    "cached_count": len(cached_results),
                },
                input_data=_summarize_for_trace(input_state),
        ) as run_span):

            graph_config = {"configurable": {"thread_id": self.config.run_id}}

            current_state = None
            for chunk in self.graph.stream(input_state, config=graph_config, stream_mode=["tasks", "updates"]): # pyrefly: ignore [no-matching-overload]
                stream_type, data = chunk
                if stream_type == "tasks":
                    yield {
                        "event": "node_started",
                        "node": data['name']
                    }
                elif stream_type == "updates":
                    node_name = list(data.keys())[0]
                    current_state = State(**data[node_name])
                    yield {"event": "node_finished", "node": node_name, "state": current_state}
            final_result = current_state
            if not final_result:
                raise Exception("The Graph was not able to produce a result")

            final_result.profiling_metrics['total_run_time_sec'] = round(time.perf_counter() - _run_t0, 3)

            if self.config.use_cache and self.config.cache_mode in ["write", "w", "read_write", "rw"]:
                with self.trace_helper.start_span(
                        "cache_save_run",
                        kind="tool",
                        input_data={"run_id": self.config.run_id, "prompt": truncate_trace_text(self.config.prompt)},
                ) as span:
                    yield {"event": "cache", "value": "store"}
                    self.cache.save_run(
                        config=self.config,
                        final_result=final_result,
                    )
                    yield {"event": "cache", "value": "store_completed"}
                    tracing.set_output(
                        span,
                        {
                            "run_id": self.config.run_id,
                            "saved": True,
                            "step_result_count": len(final_result.answers),
                        },
                    )

            tracing.set_output(run_span, _summarize_for_trace(final_result))

        # codecarbon tracker metrics
        if tracker is not None:
            tracker.stop()
            if hasattr(tracker, "final_emissions_data") and tracker.final_emissions_data is not None:
                ed = tracker.final_emissions_data
                energy_dict = {
                        "energy_consumed_kwh": ed.energy_consumed,
                        "cpu_energy_kwh": ed.cpu_energy,
                        "gpu_energy_kwh": ed.gpu_energy,
                        "ram_energy_kwh": ed.ram_energy,
                        "emissions_kg_co2": ed.emissions,
                        "cpu_power_w": ed.cpu_power,
                        "gpu_power_w": ed.gpu_power,
                        "duration_sec": ed.duration,
                }
                final_result.profiling_metrics['energy'] = energy_dict
                yield {"event": "codecarbon", "energy_dict" : energy_dict}

        if self.config.save_state:
            self._write_execution_artifacts(result_state=final_result)

        yield {"event": "completed", "state": final_result}
        return

    def check_ollama(self):
        with self.trace_helper.start_span(
                "ollama_check",
                kind="tool",
                input_data={"provider": self.config.provider, "ollama_url": self.config.ollama_url},
        ) as span:
            try:
                llm_tools.get_llm(
                    provider=self.config.provider,
                    model=self.config.model,
                    ollama_url=self.config.ollama_url
                ).invoke("Hello, how are you?")
                tracing.set_output(span, {"reachable": True})
                return True
            except Exception as e:
                tracing.set_output(span, {"reachable": False, "error": truncate_trace_text(e)})
                return False

    def check_model(self):
        """Check if the model is running locally (Ollama) or accessible (OpenAI)"""
        with self.trace_helper.start_span(
                "model_access_check",
                kind="tool",
                input_data={"provider": self.config.provider, "model": self.config.model},
        ) as span:
            if self.config.provider == "openai":
                try:
                    llm_tools.get_llm(
                        provider=self.config.provider,
                        model=self.config.model,
                    )
                    tracing.set_output(span, {"reachable": True, "provider": self.config.provider})
                    return True
                except Exception as e:
                    tracing.set_output(span, {"reachable": False, "error": truncate_trace_text(e)})
                    return False
            else:
                try:
                    base = self.config.ollama_url.rstrip("/")
                    requests.get(f"{base}/api/version", timeout=3).json()
                    reachable = self.check_ollama()
                    tracing.set_output(span, {"reachable": reachable, "provider": self.config.provider})
                    return reachable
                except Exception as e:
                    tracing.set_output(span, {"reachable": False, "error": truncate_trace_text(e)})
                    return False

    def _write_execution_artifacts(self, result_state: State):
        """Persist the execution for further analysis"""
        pass


__all__ = ["SalesDataWorkflow"]
