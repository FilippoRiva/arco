import time
from typing import Any
from typing import Generator

import requests
from langgraph.graph.state import CompiledStateGraph

from arco.core import ArcoConfig, State, llm_tools, tracking
from arco.data import RunCache


class WorkflowExecutor:
    def __init__(
            self,
            *,
            graph: CompiledStateGraph,
            config: ArcoConfig
    ) -> None:
        self.config = config
        self.graph: CompiledStateGraph = graph
        self.model_is_reachable = False

        # Updates global parameters from config
        llm_tools.OLLAMA_URL = config.ollama_url

        # Caching
        self.cache = RunCache(config.save_dir)

        # codecarbon Emission Tracking
        tracking.initialize_tracking(config)

    def stream(self) -> Generator[dict[str, Any]]:

        yield {"event": "started", "run_id": self.config.run_id, "config": self.config}

        # Global Tracking start
        tracking.start_tracking()

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

        # Check Model Reachability
        if not self.model_is_reachable:
            self.model_is_reachable = self.check_model()
            if not self.model_is_reachable:
                yield {"event": "error",
                       "message": "Model is not reachable. Please set your OPENAI_API_KEY/OPENROUTER_API_KEY environment variable if using openai/openrouter models or properly start the ollama server."}
                return None

        # Start Inference and Generator Loop
        _run_t0 = time.perf_counter()

        graph_config = {
            "configurable": {
                "thread_id": self.config.run_id,
                "enable_budget_controller": self.config.enable_budget_controller
            }
        }

        current_state = None
        final_result = None

        for chunk in self.graph.stream(input_state, config=graph_config, stream_mode=["tasks", "updates", "messages"]):
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
            elif stream_type == "messages":
                message_chunk, metadata = data
                yield {
                    "event": "token",
                    "node": metadata.get("langgraph_node"),
                    "content": message_chunk.content
                }

        final_result = current_state
        if not final_result:
            yield {"event": "error",
                   "message": "The Graph was not able to produce a result"}

        if self.config.use_cache and self.config.cache_mode in ["write", "w", "read_write", "rw"]:
            yield {"event": "cache", "value": "store"}
            self.cache.save_run(
                config=self.config,
                final_result=final_result,
            )
            yield {"event": "cache", "value": "store_completed"}

        if final_result is not None and self.config.save_state:
            final_result.save(self.config.save_dir)

        # Global tracking stop
        tracking.stop_tracking()

        yield {"event": "completed", "state": final_result}
        return final_result

    def check_ollama(self):
        try:
            llm_tools.get_llm(
                provider=self.config.provider,
                model=self.config.model,
            ).invoke("Hello, how are you?")
            return True
        except Exception as e:
            return False

    def check_model(self):
        """Check if the model is running locally (Ollama) or accessible (OpenAI)"""
        if self.config.provider == "openai" or self.config.provider == "openrouter":
            try:
                llm_tools.get_llm(
                    provider=self.config.provider,
                    model=self.config.model,
                )
                return True
            except Exception as e:
                yield {"event": "error",
                       "message": "Model is not reachable. Please set your OPENAI_API_KEY/OPENROUTER_API_KEY environment variable if using openai/openrouter models or properly start the ollama server."}
                return False
        else:
            try:
                base = self.config.ollama_url.rstrip("/")
                requests.get(f"{base}/api/version", timeout=3).json()
                reachable = self.check_ollama()
                return reachable
            except Exception as e:
                return False


__all__ = ["WorkflowExecutor"]
