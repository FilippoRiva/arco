import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import requests
from langgraph.graph.state import CompiledStateGraph

from arco.core import Config, State, llm_tools, tracking


class WorkflowExecutor:
    def __init__(self, *, workflow: Workflow, config: Config) -> None:
        self.config = config
        self.graph: CompiledStateGraph = workflow.graph
        self.model_is_reachable = False

        # Updates global parameters from config
        llm_tools.OLLAMA_URL = config.ollama_url

        # codecarbon Emission Tracking
        tracking.initialize_tracking(config)

    def stream(self) -> Generator[dict[str, Any]]:

        yield {"event": "started", "run_id": self.config.run_id, "config": self.config}

        # Global Tracking start
        tracking.start_tracking()

        # Initialize state
        input_state: State = State(
            prompt=self.config.prompt,
            run_id=self.config.run_id,
            agent_configs=self.config.agent_configs,
        )

        # Check Model Reachability
        if not self.model_is_reachable:
            self.model_is_reachable = self._check_model()
            if not self.model_is_reachable:
                yield {
                    "event": "error",
                    "message": "Model is not reachable. Please set your OPENAI_API_KEY/OPENROUTER_API_KEY environment variable if using openai/openrouter models or properly start the ollama server.",
                }
                return None

        # Start Inference and Generator Loop
        _run_t0 = time.perf_counter()

        graph_config = {
            "configurable": {
                "thread_id": self.config.run_id,
                "enable_budget_controller": self.config.enable_budget_controller,
            }
        }

        current_state = None

        for chunk in self.graph.stream(
            input_state,
            config=graph_config,
            stream_mode=["tasks", "updates", "messages"],
        ):
            stream_type, data = chunk
            if stream_type == "tasks":
                yield {"event": "node_started", "node": data["name"]}
            elif stream_type == "updates":
                node_name = list(data.keys())[0]
                current_state = State(**data[node_name])
                yield {
                    "event": "node_finished",
                    "node": node_name,
                    "state": current_state,
                }
            elif stream_type == "messages":
                message_chunk, metadata = data
                yield {
                    "event": "token",
                    "node": metadata.get("langgraph_node"),
                    "content": message_chunk.content,
                }

        final_result = current_state
        if not final_result:
            yield {
                "event": "error",
                "message": "The Graph was not able to produce a result",
            }

        if final_result is not None and self.config.enable_storage:
            final_result.save(Path(self.config.save_dir) / "storage")

        # Global tracking stop
        tracking.stop_tracking()

        yield {"event": "completed", "state": final_result}
        return final_result

    def _check_model(self):
        """Check if the model is running locally (Ollama) or accessible (OpenAI)"""
        if (
            self.config.default_provider == "openai"
            or self.config.default_provider == "openrouter"
        ):
            try:
                llm_tools.get_llm(
                    provider=self.config.default_provider,
                    model=self.config.default_model,
                )
                return True
            except Exception:
                return False
        else:
            try:
                base = self.config.ollama_url.rstrip("/")
                requests.get(f"{base}/api/version", timeout=3).json()
                reachable = self._check_ollama()
                return reachable
            except Exception:
                return False

    def _check_ollama(self):
        try:
            llm_tools.get_llm(
                provider=self.config.default_provider,
                model=self.config.default_model,
            ).invoke("Hello, how are you?")
            return True
        except Exception:
            return False
