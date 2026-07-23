from typing import override

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from arco.agents import Analyzer, Orchestrator, Planner, Retriever, Visualizer
from arco.core import Config, State
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
            if answer and 'agent_choice' in answer.agent_output and answer.agent_output[
                'agent_choice'] in valid_choices:
                return answer.agent_output['agent_choice']
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


class PlannedSales(Workflow):
    workflow_id = "planned_sales"

    @override
    def _initialize(self, config: Config) -> CompiledStateGraph:
        graph = StateGraph(State)

        planner = Planner()
        retriever = Retriever()
        analyzer = Analyzer()
        visualizer = Visualizer()

        for agent in [planner, retriever, analyzer, visualizer]:
            graph.add_node(agent.name, agent)
            self._agent_list.update({agent.type: agent})

        graph.set_entry_point(planner.name)

        def route_from_plan(state: State) -> str:
            answer = state.get_last_answer(planner.type)
            valid_choices = [
                retriever.name,
                analyzer.name,
                visualizer.name,
            ]
            if answer and "agent_choice" in answer.agent_output and answer.agent_output[
                "agent_choice"
            ] in valid_choices:
                return answer.agent_output["agent_choice"]
            return "end"

        graph.add_conditional_edges(
            planner.name,
            route_from_plan,
            {
                retriever.name: retriever.name,
                analyzer.name: analyzer.name,
                visualizer.name: visualizer.name,
                "end": END,
            },
        )

        graph.add_edge(retriever.name, planner.name)
        graph.add_edge(analyzer.name, planner.name)
        graph.add_edge(visualizer.name, planner.name)

        return graph.compile()
