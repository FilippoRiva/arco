import time
from unittest.mock import Mock

import pytest

from arco.core import AgentType, Answer
from arco.core.config import AgentConfig
from arco.core.state import State


@pytest.fixture()
def agent_config():
    return AgentConfig(agent_name="test_agent_config")

@pytest.fixture
def base_state_with_mock(agent_config):
    return State(
        prompt="Generate a bar chart",
        run_id="test_run_123",
        agent_configs={AgentType.ANALYZER: agent_config, AgentType.VISUALIZER: agent_config},
    )

@pytest.fixture
def first_answer(agent_config):
    return Answer(
        agent_id=AgentType.ANALYZER,
        message="This is the Answer's message",
        agent_config = agent_config
    )

@pytest.fixture
def second_answer(agent_config):
    return Answer(
        agent_id=AgentType.VISUALIZER,
        message="This is the visualization code : import stuff",
        agent_config = agent_config
    )

@pytest.fixture
def full_answer(agent_config):
    return Answer(
        agent_id=AgentType.ANALYZER,
        message="This is the Answer's message",
        evaluation=Mock(),
        gt_evaluation=Mock(),
        agent_choice=AgentType.VISUALIZER,
        data_str="test_string",
        data_df=Mock(),
        sql_query="SELECT * from table",
        analysis="In depth data analysis",
        chart_config={"test":"chart","config":0},
        code="import matplotlib",
        error="fatal error",
        agent_config=agent_config
    )

def test_state_initialization(agent_config):
    base_state =  State(
        prompt="Generate a bar chart",
        run_id="test_run_123",
        agent_configs={AgentType.ANALYZER: agent_config, AgentType.VISUALIZER: agent_config},
    )

    config_analyzer = base_state.get_agent_config(AgentType.ANALYZER)
    config_retriever = base_state.get_agent_config(AgentType.RETRIEVER)
    config_visualizer = base_state.get_agent_config(AgentType.VISUALIZER)

    assert config_analyzer, "I should be able to find the ANALYZER config"
    assert config_visualizer, "VISUALIZER configs should be present"
    assert not config_retriever, "I should not find RETRIEVER configs"
    assert len(base_state.answers) == 0

def test_answer_management(base_state_with_mock, first_answer, second_answer):
    middle_state = base_state_with_mock.add_answer(first_answer)
    middle_answer = middle_state.get_last_answer()
    final_state = middle_state.add_answer(second_answer)
    last_answer = final_state.get_last_answer()

    last_analyzer_answer = final_state.get_last_answer(AgentType.ANALYZER)
    last_visualizer_answer = final_state.get_last_answer(AgentType.VISUALIZER)

    last_analyzer_config = final_state.get_last_agent_config(AgentType.ANALYZER)
    last_visualizer_config = final_state.get_last_agent_config(AgentType.VISUALIZER)
    last_config_no_type = final_state.get_last_agent_config()

    (middle_tuple_ans, middle_tuple_config) = middle_state.get_last_execution_outputs()
    (last_tuple_ans, last_tuple_config) = final_state.get_last_execution_outputs()

    assert middle_state != base_state_with_mock, "State should be frozen"
    assert len(middle_state.answers) == 1, "State should contain 1 answer"
    assert middle_answer == first_answer
    assert middle_answer == last_analyzer_answer

    assert final_state != base_state_with_mock, "State should be frozen"
    assert len(final_state.answers) == 2, "State should contain 2 answers"
    assert last_answer == second_answer, "Last answer should be the last added answer"
    assert last_answer == last_visualizer_answer

    assert middle_answer != last_answer

    assert last_analyzer_config is base_state_with_mock.agent_configs[AgentType.ANALYZER], "Agent_configs should not mutate"
    assert last_visualizer_config is base_state_with_mock.agent_configs[AgentType.VISUALIZER], "Agent_configs should not mutate"

    assert last_config_no_type == last_visualizer_config, "The last no type config should be the one of the last agent"

    assert middle_tuple_ans == middle_answer
    assert middle_tuple_config == last_analyzer_config
    assert last_tuple_ans == last_answer
    assert last_tuple_config == last_visualizer_config

def test_answer_to_string(full_answer):
    result = full_answer.__str__()
    assert result, "representation should not be None"

def test_answers_stringify(base_state_with_mock, first_answer):
    empty_result = base_state_with_mock.stringify_answers(max_message_length=10)
    updated_state = base_state_with_mock.add_answer(first_answer)
    full_result = updated_state.stringify_answers()
    max_length = 5
    stripped_result = updated_state.stringify_answers(max_message_length=max_length)

    assert empty_result == "", "No str should return when no answers are set"
    assert first_answer.message in full_result, "The full message should be contained in the representation string"
    assert len(stripped_result) == (len(full_result) - len(first_answer.message)) + 5 , "The returned message should be limited in length if requested"

def test_profiling_metrics(base_state_with_mock):
    total_timings = 1
    start_counter = time.perf_counter()
    agent_type = AgentType.ANALYZER
    energy = {
        "test_dict": "with_no_meaning"
    }
    result = base_state_with_mock.set_profiling_metrics(total_timings, start_counter, agent_type, energy)

    assert result, "Shouldn't be None"
    assert result != base_state_with_mock, "Should be a different object"
    assert result.global_profiling_data, "Shouldn't be None"

def test_dict(base_state_with_mock, first_answer):
    answer_dict = first_answer.to_dict()
    answer_reconstruction = Answer.from_dict(answer_dict)

    state_dict = base_state_with_mock.to_dict()
    state_reconstruction = State.from_dict(state_dict)

    assert answer_reconstruction == first_answer
    assert state_reconstruction == base_state_with_mock
