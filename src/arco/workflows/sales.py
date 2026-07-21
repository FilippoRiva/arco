from typing import override

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from arco.core import Config, State
from arco.agents import Analyzer, Orchestrator, Retriever, Visualizer
from arco.workflows.workflow import Workflow


class StrictSales(Workflow):
    workflow_id = "sales"

    @override
    def _initialize(self, config: Config) -> CompiledStateGraph:
        graph = StateGraph(State)

        # Get Agents
        retriever = Retriever()
        analyzer = Analyzer()
        visualizer = Visualizer()

        # Add nodes
        for agent in [retriever, analyzer, visualizer]:
            graph.add_node(agent.name, agent)
            self._agent_list.update({agent.type: agent})

        # Add entry point
        graph.set_entry_point(retriever.name)

        # Add edges
        graph.add_edge(retriever.name, analyzer.name)
        graph.add_edge(analyzer.name, visualizer.name)
        graph.add_edge(visualizer.name, END)

        return graph.compile()


class OrchestratedSales(Workflow):
    workflow_id = "orchestrated_sales"

    @override
    def _initialize(self, config: Config) -> CompiledStateGraph:
        graph = StateGraph(State)

        # Get agents
        orchestrator = Orchestrator()
        retriever = Retriever()
        analyzer = Analyzer()
        visualizer = Visualizer()

        # Add nodes
        for agent in [orchestrator, retriever, analyzer, visualizer]:
            graph.add_node(agent.name, agent)
            self._agent_list.update({agent.type: agent})

        # Entry point
        graph.set_entry_point(orchestrator.name)

        def route_to_agent(state: State) -> str:
            answer = state.get_last_answer(orchestrator.type)
            valid_choices = [
                retriever.name,
                analyzer.name,
                visualizer.name
            ]
            if answer and answer.agent_choice and answer.agent_choice in valid_choices:
                return answer.agent_choice
            return "end"

        # Routing logic
        graph.add_conditional_edges(
            orchestrator.name,
            route_to_agent,
            {
                retriever.name: retriever.name,
                analyzer.name: analyzer.name,
                visualizer.name: visualizer.name,
                "end": END,
            },
        )

        # Edges returning to orchestrator
        graph.add_edge(retriever.name, orchestrator.name)
        graph.add_edge(analyzer.name, orchestrator.name)
        graph.add_edge(visualizer.name, orchestrator.name)

        return graph.compile()

