import logging
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from langgraph.graph.state import CompiledStateGraph

from arco.core import Config, State, llm_tools, tracking

if TYPE_CHECKING:
    from arco.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


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
        requested_models = [
            *[
                (agent_config.provider, agent_config.model)
                for agent_config in self.config.agent_configs.values()
            ],
            *[
                (agent_config.provider_judge, agent_config.model_judge)
                for agent_config in self.config.agent_configs.values()
            ],
        ]
        unique_models = list(set(requested_models))
        yield {"event": "check_connection", "models": unique_models}
        for provider, model in unique_models:
            self.model_is_reachable, message = self._check_model(
                provider=provider, model=model
            )
            if not self.model_is_reachable:
                yield {"event": "error", "message": message}
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
                node_name = next(iter(data.keys()))
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

    def _check_model(self, provider: str, model: str) -> tuple[bool, str]:
        """Check if the configured LLM provider is reachable.

        Uses lightweight probes — no inference calls are made.
        """
        if provider in ("openai", "openrouter"):
            import openai

            try:
                api_key = os.environ.get(
                    "OPENROUTER_API_KEY"
                    if provider == "openrouter"
                    else "OPENAI_API_KEY"
                )
                if not api_key:
                    error_message = f"Missing API key for {provider}. Set the {'OPENROUTER_API_KEY' if provider == 'openrouter' else 'OPENAI_API_KEY'} environment variable."
                    logger.error(error_message)
                    return False, error_message

                models = []
                if provider == "openai":
                    models = openai.OpenAI(api_key=api_key, timeout=5.0).models.list()
                elif provider == "openrouter":
                    models = openai.OpenAI(
                        api_key=api_key,
                        timeout=5.0,
                        base_url="https://openrouter.ai/api/v1",
                    ).models.list()

                models = [provider_model.id for provider_model in models]
                if model not in models:
                    raise ValueError(
                        f"The requested model is not available: '{model}'. Available models are {models}"
                    )
                return True, f"Connection to {provider} succeeded."
            except openai.OpenAIError as e:
                error_message = f"{provider} connection failed: {e}"
                logger.error(error_message)
                return False, error_message
            except ValueError as e:
                error_message = f"{provider} connection failed: {e}"
                logger.error(error_message)
                return False, error_message

        # Ollama
        try:
            base = self.config.ollama_url.rstrip("/")
            resp = requests.get(f"{base}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = [
                model.get("model").split(":")[0] for model in resp.json()["models"]
            ]
            if model.split(":")[0] not in models:
                raise ValueError(
                    f"The requested model is not available: '{model}'. Available models are {models}"
                )
            return True, f"Connection to {provider} succeeded."
        except requests.RequestException as e:
            error_message = f"{provider} connection failed: {e}"
            logger.error(error_message)
            return False, error_message
        except ValueError as e:
            error_message = f"{provider} connection failed: {e}"
            logger.error(error_message)
            return False, error_message
