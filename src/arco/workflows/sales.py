from typing import override

from arco.agents import Analyzer, Orchestrator, Planner, Retriever, Visualizer
from arco.core import Agent, Config, Graph, State, Workflow
from arco.core.graph import END


class StrictSales(Workflow):
    workflow_id = "sales"

    @override
    def initialize(self, config: Config, graph: Graph):
        # Get Agents
        retriever = Retriever()
        analyzer = Analyzer()
        visualizer = Visualizer()

        # Add nodes
        for agent in [retriever, analyzer, visualizer]:
            graph.add_agent(agent)

        # Add entry point
        graph.set_entry_agent(retriever)

        # Add edges
        graph.add_agent_edge(retriever, analyzer)
        graph.add_agent_edge(analyzer, visualizer)
        graph.add_agent_edge(visualizer, END)


def _instrument_orchestrated_graph(graph: Graph, orchestrating_agent: Agent):
    retriever = Retriever()
    analyzer = Analyzer()
    visualizer = Visualizer()

    # Add nodes
    for agent in [orchestrating_agent, retriever, analyzer, visualizer]:
        graph.add_agent(agent)

    # Entry point
    graph.set_entry_agent(orchestrating_agent)

    def route_to_agent(state: State) -> str:
        answer = state.get_last_answer(orchestrating_agent.type)
        if answer and "agent_choice" in answer.agent_output:
            return answer.agent_output["agent_choice"]
        return "End"

    # Routing logic
    graph.add_conditional_edges(
        orchestrating_agent,
        route_to_agent,
        {
            retriever: retriever,
            analyzer: analyzer,
            visualizer: visualizer,
            "End": END,
        },
    )

    # Edges returning to orchestrator
    graph.add_agent_edge(retriever, orchestrating_agent)
    graph.add_agent_edge(analyzer, orchestrating_agent)
    graph.add_agent_edge(visualizer, orchestrating_agent)


class OrchestratedSales(Workflow):
    workflow_id = "orchestrated_sales"

    @override
    def initialize(self, config: Config, graph: Graph):
        orchestrator = Orchestrator()
        _instrument_orchestrated_graph(graph=graph, orchestrating_agent=orchestrator)


class PlannedSales(Workflow):
    workflow_id = "planned_sales"

    @override
    def initialize(self, config: Config, graph: Graph):
        planner = Planner()
        _instrument_orchestrated_graph(graph=graph, orchestrating_agent=planner)


__all__ = ["OrchestratedSales", "PlannedSales", "StrictSales"]
