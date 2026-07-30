from collections.abc import Awaitable, Callable, Hashable, Sequence
from typing import Self

from langgraph.graph import END as LANGGRAPH_END
from langgraph.graph import StateGraph

END = LANGGRAPH_END

from .agent import Agent
from .agent_type import AgentType
from .state import State


class Graph(StateGraph):
    """A :class:`StateGraph` subclass that accepts :class:`Agent` instances
    directly in place of node/edge names.

    When an :class:`Agent` is passed to :meth:`add_node`, the agent's
    ``name`` is used as the node name and the agent is stored for later
    retrieval via :meth:`get_agents`.
    """

    def __init__(self):
        super().__init__(State)
        self._agents: dict[AgentType, Agent] = {}

    def add_agent(
        self,
        node: str | Agent,
        action: Agent | None = None,
    ) -> Self:
        """Add a node to the graph.

        If *node* is an :class:`Agent`, its ``name`` is used as the
        node name and the agent is stored in the internal registry.
        """
        if isinstance(node, Agent):
            self._agents.update({node.type: node})
            action = node
            node = node.name
        super().add_node(node, action)

    def set_entry_agent(self, agent: Agent) -> Self:
        """Set the entry point of the graph.

        Accepts an :class:`Agent` instance or a string node name.
        """
        if isinstance(agent, Agent):
            agent = agent.name
        super().set_entry_point(agent)

    def add_agent_edge(self, from_node: Agent | str | list[str], to_node: Agent | str):
        """Add a directed edge between two nodes.

        Accepts :class:`Agent` instances or string node names.
        """
        if isinstance(from_node, Agent):
            from_node = from_node.name
        if isinstance(to_node, Agent):
            to_node = to_node.name
        super().add_edge(from_node, to_node)

    def get_agents(self) -> dict[AgentType, Agent]:
        """Return a copy of the internal agent registry."""
        return self._agents.copy()

    def add_conditional_edges(
        self,
        source: Agent | str,
        path: Callable[..., Hashable | Sequence[Hashable]]
        | Callable[..., Awaitable[Hashable | Sequence[Hashable]]],
        path_map: dict[Hashable, str] | list[str] | None = None,
    ) -> Self:
        """Add conditional edges from a node.

        Accepts :class:`Agent` instances in *source* and in the
        keys and values of *path_map*.
        """
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
