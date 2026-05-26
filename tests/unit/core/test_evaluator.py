import pytest

from arco.core import Evaluator, State, Answer, AgentType, Evaluation
from unittest.mock import Mock

@pytest.fixture
def config():
    return Mock()

# noinspection PyUnresolvedReferences
def test_evaluator(config):
    evaluator = Evaluator()
    states = [
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_a", agent_config=config),
            Answer(agent_id=AgentType.ANALYZER, message="second_answer_a", agent_config=config),
            Answer(agent_id=AgentType.VISUALIZER, message="third_answer_a", agent_config=config),
        ]),
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_b", agent_config=config),
            Answer(agent_id=AgentType.ANALYZER, message="second_answer_b", agent_config=config),
            Answer(agent_id=AgentType.VISUALIZER, message="third_answer_b", agent_config=config),
        ]),
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_c", agent_config=config),
            Answer(agent_id=AgentType.ANALYZER, message="second_answer_c", agent_config=config),
            Answer(agent_id=AgentType.VISUALIZER, message="third_answer_c", agent_config=config),
        ]),
    ]

    (states, selected_state) = evaluator.evaluate_and_select(states)

    assert selected_state is not None
    assert selected_state.get_last_answer().evaluation is not None
    for state in states:
        assert state.get_last_answer().evaluation is not None
        assert state.get_last_answer().evaluation.success == False

def test_default_evaluator_selection(config):
    evaluator = Evaluator()

    best_state = State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={},
                       answers=[Answer(agent_id=AgentType.RETRIEVER, agent_config=config,
                                       message="first_answer_a", evaluation=Evaluation(score=1))])

    states_with_fake_evaluations = [
        best_state,
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_a",
                   evaluation=Evaluation(score=0.8), agent_config=config),
        ]),
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_a",
                   evaluation=Evaluation(score=0.85), agent_config=config),
        ]),
    ]

    selection = evaluator._selection(states_with_fake_evaluations)

    states_with_no_evaluation = [
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_a",
                   evaluation=None, agent_config=config),
        ]),
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_a",
                   evaluation=None, agent_config=config),
        ]),
    ]

    second_selection = evaluator._selection(states_with_no_evaluation)

    assert selection == best_state
    assert second_selection is not None

# noinspection PyUnresolvedReferences
def test_default_evaluator_gt_evaluation(config):
    evaluator = Evaluator()

    states = [
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_a", agent_config=config)]),
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_b", agent_config=config)]),
        State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={}, answers=[
            Answer(agent_id=AgentType.RETRIEVER, message="first_answer_c", agent_config=config)]),
    ]

    evaluator.evaluate_ground_truth(states)

    for state in states:
        assert state.get_last_answer().gt_evaluation is not None
        assert state.get_last_answer().gt_evaluation.success == False

def test_no_answers_eval():
    evaluator = Evaluator()
    state = State(prompt="First prompt", visualization_goal="Vis goal", run_id="1", agent_configs={},
                  answers=[])

    with pytest.raises(ValueError):
        evaluator._eval(state)