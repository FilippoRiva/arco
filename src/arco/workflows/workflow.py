import inspect
from abc import abstractmethod, ABC
import inspect

from langgraph.graph.state import CompiledStateGraph

from arco.core import Config, AgentType, Agent


class Workflow(ABC):
    _registry: dict[str, type[Workflow]] = {}
    _agent_list: dict[AgentType, Agent] = {}

    def __init_subclass__(cls, **kwargs):
        """When a subclass inherits this ABC, the workflow_id of that subclass is stored and the WorkflowFactory can
        retrieve a new instance of that Workflow from the workflow_id itself. This provides compatibility with
        any kind of dynamically defined workflow whenever it inherits from this ABC"""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # don't register intermediate abstract subclasses
        id = getattr(cls, "workflow_id", cls.__name__)
        if id in Workflow._registry and Workflow._registry[id] is not cls:
            raise TypeError(f"Workflow id {id!r} already registered to {Workflow._registry[id]!r}")
        Workflow._registry[id] = cls

    def __init__(self, config: Config):
        self.graph : CompiledStateGraph = self._initialize(config)

    @abstractmethod
    def _initialize(self, config: Config) -> CompiledStateGraph:
        ...

    @classmethod
    def get(cls, name: str) -> type[Workflow]:
        try:
            return cls._registry[name]
        except KeyError:
            raise ValueError(
                f"Unknown workflow {name!r}. Available: {sorted(cls._registry)}"
            ) from None

    @classmethod
    def all(cls) -> dict[str, type[Workflow]]:
        return dict(cls._registry)

    def get_agent(self, agent_type: AgentType) -> Agent:
        return self._agent_list[agent_type]

    def get_evaluators(self) -> dict[AgentType, Evaluator]:
        return {agent_type: agent.get_evaluator() for agent_type, agent in self._agent_list.items()}

    def __str__(self) -> str:
        return self.graph.get_graph().draw_ascii()


class WorkflowFactory:
    @staticmethod
    def get_from_config(config_path: str) -> tuple[Workflow, Config]:
        config = Config.from_yaml(config_path)
        workflow = Workflow.get(config.workflow)(config)
        config.hydrate_agent_configs(config_path)
        return workflow, config
