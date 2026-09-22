from typing import TYPE_CHECKING, override

from arco.core import Workflow

if TYPE_CHECKING:
    from arco.core import Agent, Config, Graph, State


class StrictSales(Workflow):
    workflow_id = "strict_sales"
    description = (
        "Runs the sales pipeline in a fixed order: retrieval, analysis, then visualization."
    )

    @override
    def initialize(self, config: Config, graph: Graph):
        from arco.agents import Analyzer, Retriever, Visualizer
        from arco.core.graph import END

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
    from arco.agents import Analyzer, Retriever, Visualizer
    from arco.core import State  # noqa: F401 - needed by langgraph
    from arco.core.graph import END

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

    path_map: dict[str | Agent, Agent | str] = {
        retriever: retriever,
        analyzer: analyzer,
        visualizer: visualizer,
        "End": END,
    }

    # Routing logic
    graph.add_conditional_edges(
        source=orchestrating_agent, path=route_to_agent, path_map=path_map
    )

    # Edges returning to orchestrator
    graph.add_agent_edge(retriever, orchestrating_agent)
    graph.add_agent_edge(analyzer, orchestrating_agent)
    graph.add_agent_edge(visualizer, orchestrating_agent)


class OrchestratedSales(Workflow):
    workflow_id = "orchestrated_sales"
    description = (
        "Uses an LLM orchestrator to choose the next sales-analysis agent dynamically."
    )

    @override
    def initialize(self, config: Config, graph: Graph):
        from arco.agents import Orchestrator

        orchestrator = Orchestrator()
        _instrument_orchestrated_graph(graph=graph, orchestrating_agent=orchestrator)


class PlannedSales(Workflow):
    workflow_id = "planned_sales"
    description = (
        "Uses an upfront LLM-generated plan to run the required sales-analysis agents."
    )

    @override
    def initialize(self, config: Config, graph: Graph):
        from arco.agents import Planner

        planner = Planner()
        _instrument_orchestrated_graph(graph=graph, orchestrating_agent=planner)


__all__ = ["OrchestratedSales", "PlannedSales", "StrictSales"]
