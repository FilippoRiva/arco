import inspect
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.graph.state import CompiledStateGraph

from arco.core import Agent, AgentType, Config, State, llm_tools, tracking
from arco.core.graph import Graph

if TYPE_CHECKING:
    from arco.core import Evaluator

logger = logging.getLogger(__name__)


class Workflow(ABC):
    def __init_subclass__(cls, **kwargs):
        """When a subclass inherits this ABC, the workflow_id of that subclass is stored and the WorkflowFactory can
        retrieve a new instance of that Workflow from the workflow_id itself. This provides compatibility with
        any kind of dynamically defined workflow whenever it inherits from this ABC"""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # don't register intermediate abstract subclasses
        workflow_id = getattr(cls, "workflow_id", cls.__name__.lower())
        WorkflowFactory._workflow_registry[workflow_id] = cls

    def __init__(self, config: Config | None = None):
        if self.workflow_id is None:
            self.workflow_id = self.__class__.__name__.lower()
        self._agent_list: dict[AgentType, Agent] = {}
        self.config = (
            config.set(workflow=self.workflow_id)
            if config
            else Config(workflow=self.workflow_id)
        )
        self.graph: CompiledStateGraph = self._initialize(self.config)
        self.config.hydrate_agent_configs(self.list_agents())

    def stream(self, config: Config | None = None) -> Generator[dict[str, Any]]:

        yield {"event": "started", "run_id": self.config.run_id, "config": self.config}

        if config:
            self.config = config
            self.config.hydrate_agent_configs(self.list_agents())

        # Updates global parameters from config
        llm_tools.OLLAMA_URL = self.config.ollama_url

        # codecarbon Emission Tracking
        tracking.initialize_tracking(self.config)

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
            reachable, message = llm_tools.check_model_availability(
                provider=provider, model=model
            )
            if not reachable:
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
            if stream_type == "tasks" and "input" in data:
                yield {"event": "node_started", "node": data["name"]}
                logger.debug("node_started_event: " + str(data))
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
            logger.error("No final result has been retrieved from the graph stream.")
            yield {
                "event": "error",
                "message": "The Graph was not able to produce a result",
            }

        if final_result is not None and self.config.enable_storage:
            final_result.save(Path(self.config.save_dir) / "storage")

        yield {"event": "completed", "state": final_result}
        logger.info("Workflow completed successfully")
        return final_result

    @abstractmethod
    def initialize(self, config: Config, graph: Graph):
        """Given a Config and an empty Graph, builds the workflow graph, depending on implementation"""
        ...

    def _initialize(self, config: Config) -> CompiledStateGraph:
        graph = Graph()
        self.initialize(config, graph)
        self._agent_list.update(graph.get_agents())
        return graph.compile()

    def list_agents(self) -> list[AgentType]:
        return list(self._agent_list.keys())

    def get_agent(self, agent_type: AgentType) -> Agent:
        return self._agent_list[agent_type]

    def get_evaluators(self) -> dict[AgentType, Evaluator]:
        return {
            agent_type: agent.evaluator
            for agent_type, agent in self._agent_list.items()
        }

    def __str__(self) -> str:
        return self.graph.get_graph().draw_ascii()


class WorkflowFactory:
    _workflow_registry: ClassVar[dict[str, type[Workflow]]] = {}

    @classmethod
    def all(cls) -> dict[str, type[Workflow]]:
        return dict(cls._workflow_registry)

    @classmethod
    def get(
        cls, config: Config | None = None, workflow_id: str | None = None
    ) -> Workflow:
        if config is not None:
            workflow_cls = cls._get_cls(config.workflow)
            return workflow_cls(config)
        elif workflow_id is not None:
            workflow_cls = cls._get_cls(workflow_id)
            return workflow_cls()
        else:
            raise KeyError(
                "Either a Config or workflow_id is require to retrieve a workflow"
            )

    @classmethod
    def _get_cls(cls, workflow_id: str) -> type[Workflow]:
        try:
            return cls._workflow_registry[workflow_id]
        except KeyError:
            raise ValueError(
                f"Unknown workflow {workflow_id!r}. Available: {sorted(cls._workflow_registry)}"
            )


__all__ = ["Workflow", "WorkflowFactory"]
