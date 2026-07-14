from langgraph.graph import StateGraph, END

from arco.core import ArcoConfig, State
from arco.agents import Analyzer, Orchestrator, Retriever, Visualizer

def _strict_graph(config : ArcoConfig) -> CompiledStateGraph:
    graph = StateGraph(State)

    init_args = {
        "empower": config.empower
    }

    # Get Agents
    retriever = Retriever(schema=config.schema, **init_args)
    analyzer = Analyzer(**init_args)
    visualizer = Visualizer(**init_args)

    # Add nodes
    for agent in [retriever, analyzer, visualizer]:
        graph.add_node(agent.name, agent)

    # Add entry point
    graph.set_entry_point(retriever.name)

    # Add edges
    graph.add_edge(retriever.name, analyzer.name)
    graph.add_edge(analyzer.name, visualizer.name)
    graph.add_edge(visualizer.name, END)

    return graph.compile()


def _orchestration_graph(config: ArcoConfig) -> CompiledStateGraph:
    graph = StateGraph(State)

    init_args = {
        "empower": config.empower
    }

    # Get agents
    orchestrator = Orchestrator(**init_args)
    retriever = Retriever(schema=config.schema, **init_args)
    analyzer = Analyzer(**init_args)
    visualizer = Visualizer(**init_args)

    # Add nodes
    for agent in [orchestrator, retriever, analyzer, visualizer]:
        graph.add_node(agent.name, agent)

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


def build_graph(config : ArcoConfig) -> CompiledStateGraph:
    if config.orchestration_enabled:
        return _orchestration_graph(config)
    return _strict_graph(config)

