from typing import Literal, Any
import math
import sys
from dataclasses import dataclass, fields
from typing import cast

from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

from .state import AgentType, State, Answer

@dataclass
class EmpoweredAnswer(Answer):
    perplexity: float = 0.0
    budget_controller_choice: Literal["rollback", "end"] = "end"

    @classmethod
    def from_answer(cls, answer: Answer) -> EmpoweredAnswer:
        return cls(**answer.__dict__.copy())

    @classmethod
    def from_dict(cls, dictionary: dict[str, Any]) -> EmpoweredAnswer:
        perplexity = dictionary['perplexity']
        budget_controller_choice = dictionary['budget_controller_choice']
        dictionary.pop('perplexity')
        dictionary.pop('budget_controller_choice')
        answer = Answer.from_dict(dictionary)
        empowered_ans = EmpoweredAnswer.from_answer(answer)
        empowered_ans.perplexity = perplexity
        empowered_ans.budget_controller_choice = budget_controller_choice
        return empowered_ans

    def copy(self)-> Answer:
        ## Exploits the Answer.from_dict() so empowered fields must be removed and re-added later on
        # Extract field names
        answer_field_names = {f.name for f in fields(Answer)}
        empowered_field_names = {f.name for f in fields(EmpoweredAnswer)}
        empowered_field_names = empowered_field_names - answer_field_names

        # To dict
        self_dict = self.to_dict()

        # Remove empowered from dict
        for field in empowered_field_names:
            self_dict.pop(field)
        ans = Answer.from_dict(self_dict)
        ans = EmpoweredAnswer.from_answer(ans)

        # Fix everything else
        for field in empowered_field_names:
            ans.__setattr__(field, self.__getattribute__(field))
        return ans


def get_parent_node(config: RunnableConfig) -> AgentType:
    configurable = config.get("configurable", {})
    upper_node = configurable.get("parent_node_name", "unknown_parent")
    return AgentType(upper_node)


def arco_evaluation(state: State, config: RunnableConfig) -> State:
    parent_node = get_parent_node(config)

    # Fetch the last answer object using your parent node identifier
    answer = state.get_last_answer(parent_node)
    if not answer:
        raise Exception("No answer found during ARCO evaluation")
    answer = EmpoweredAnswer.from_answer(answer)
    if answer.logprobs is None or len(answer.logprobs) == 0:
        answer.perplexity = 0.0
        return state.replace_last_answer(answer)

    # Compute Perplexity
    logprobs: list[float | int] = answer.logprobs
    avg_logprob = sum(logprobs) / len(logprobs)
    if avg_logprob < -math.log(sys.float_info.max):
        perplexity = math.inf
    else:
        perplexity = math.exp(-avg_logprob)

    # Return to EmpoweredState
    answer.perplexity=perplexity
    return state.replace_last_answer(answer)


def budget_controller(state: State, config: RunnableConfig) -> State:
    parent_node = get_parent_node(config)
    answer = state.get_last_answer(parent_node)
    if not answer:
        raise Exception("No answer found during budget controller")
    answer = cast(EmpoweredAnswer, answer)

    if parent_node == AgentType.RETRIEVER:
        max_perplexity = 2
    elif parent_node == AgentType.ANALYZER:
        max_perplexity = 15.0
    elif parent_node == AgentType.VISUALIZER:
        max_perplexity = 3
    else :
        raise Exception(f"The Budget Controller does not implement answer evaluation for this type of Agent : {parent_node.value}")

    if answer.perplexity > max_perplexity:
        answer.budget_controller_choice = "rollback"

        agent_config = state.get_agent_config(parent_node)
        agent_config.temp_min = agent_config.temp_min * 0.9
        agent_config.temp_max = agent_config.temp_max * 0.95
        if agent_config.n < 3:
            agent_config.n = agent_config.n + 1

        return state

    answer.budget_controller_choice = "end"
    return state


def budget_routing_logic(state: State, config: RunnableConfig) -> Literal["end", "rollback"]:
    parent_node = get_parent_node(config)
    answer = cast(EmpoweredAnswer, state.get_last_answer(parent_node))

    #default logic
    if not answer.budget_controller_choice or answer.budget_controller_choice not in ["end", "rollback"]:
        return "end"
    #return end or rollback
    return answer.budget_controller_choice


def empower(node_func: Runnable[State, State]) -> Runnable[State, State]:
    subgraph_builder: StateGraph = StateGraph(State)

    subgraph_builder.add_node("core_logic", node_func)
    subgraph_builder.add_node("arco_evaluation", RunnableLambda(arco_evaluation))
    subgraph_builder.add_node("budget_controller", RunnableLambda(budget_controller))

    subgraph_builder.set_entry_point("core_logic")
    subgraph_builder.add_edge("core_logic", "arco_evaluation")
    subgraph_builder.add_edge("arco_evaluation", "budget_controller")
    subgraph_builder.add_conditional_edges(
        "budget_controller",
        budget_routing_logic,
        {
            "rollback": "core_logic",
            "end": END
        },
    )

    compiled_subgraph = subgraph_builder.compile()

    def subgraph_node(state: State, config: RunnableConfig) -> State:
        metadata = config.get("metadata", {})
        upper_node = metadata.get("langgraph_node", "Unknown_node")
        if 'configurable' not in config:
            config['configurable'] = {}

        config['configurable']['parent_node_name'] = upper_node
        return cast(State, compiled_subgraph.invoke(state, config))

    return RunnableLambda(subgraph_node)
