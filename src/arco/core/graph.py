from collections.abc import Awaitable, Callable, Hashable, Sequence
from typing import Any, Self

from langgraph.graph import END as LANGGRAPH_END
from langgraph.graph import StateGraph

END = LANGGRAPH_END

from .agent import Agent
from .agent_type import AgentType
from .state import State


class Graph(StateGraph):
    def __init__(self):
        super().__init__(State)
        self._agents: dict[AgentType, Agent] = {}

    def add_node(
        self,
        node: str | Agent,
        action: Agent | None = None,
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema=None,
        retry_policy=None,
        cache_policy=None,
        error_handler=None,
        destinations=None,
        timeout=None,
    ) -> Self:
        if isinstance(node, Agent):
            self._agents.update({node.type: node})
            action = node
            node = node.name
        super().add_node(
            node,
            action,
            defer=defer,
            metadata=metadata,
            input_schema=input_schema,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            error_handler=error_handler,
            destinations=destinations,
            timeout=timeout,
        )

    def set_entry_point(self, entry_point: Agent | str) -> Self:
        if isinstance(entry_point, Agent):
            entry_point = entry_point.name
        super().set_entry_point(entry_point)

    def add_edge(self, from_node: Agent | str | list[str], to_node: Agent | str):
        if isinstance(from_node, Agent):
            from_node = from_node.name
        if isinstance(to_node, Agent):
            to_node = to_node.name
        super().add_edge(from_node, to_node)

    def get_agents(self) -> dict[AgentType, Agent]:
        return self._agents.copy()

    def add_conditional_edges(
        self,
        source: Agent | str,
        path: Callable[..., Hashable | Sequence[Hashable]]
        | Callable[..., Awaitable[Hashable | Sequence[Hashable]]],
        path_map: dict[Hashable, str] | list[str] | None = None,
    ) -> Self:
        if isinstance(source, Agent):
            source = source.name
        if isinstance(path_map, dict):
            temp = {}
            for key, value in path_map.items():
                if isinstance(key, Agent):
                    key = key.name
                if isinstance(value, Agent):
                    value = value.name
                temp.update({key: value})
            path_map = temp
        super().add_conditional_edges(source, path, path_map)


__all__ = ["END", "Graph"]
