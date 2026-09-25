from arco.core import AgentConfig, AgentType, Answer, Evaluation, Evaluator, State
from arco.core.evaluator import evaluate_state_with_benchmark_entry
from arco.data import BenchmarkEntry, Trace, TraceElement


class RecordingEvaluator(Evaluator):
    def __init__(self):
        self.evaluated: list[str] = []

    def _eval(self, state, judge_provider, judge_model, llm_accumulator=None):
        return Evaluation(score=0.0)

    def _batch_eval(self, states):
        return None

    def _gt_eval(self, answer, gt_data, judge_provider, judge_model):
        self.evaluated.append(str(answer.agent_id))
        return Evaluation(score=1.0)


def _answer(agent_name: str) -> Answer:
    return Answer(
        agent_id=AgentType(agent_name),
        message=agent_name,
        agent_config=AgentConfig(),
        logprobs=[],
    )


def test_benchmark_completion_counts_only_the_matching_trace_prefix():
    evaluator = RecordingEvaluator()
    agents = [AgentType("A"), AgentType("B"), AgentType("C")]
    entry = BenchmarkEntry(
        prompt="test",
        id=1,
        difficulty=1,
        trace=Trace(
            [TraceElement(agent_type=agent, data={}) for agent in agents]
        ),
    )
    state = State(
        prompt="test",
        run_id="test-run",
        agent_configs={},
        answers=(_answer("A"), _answer("C"), _answer("B")),
    )

    summary, updated_state = evaluate_state_with_benchmark_entry(
        state=state,
        entry=entry,
        evaluators={agent: evaluator for agent in agents},
        judge_provider="openai",
        judge_model="test",
    )

    assert summary.completion_percentage == 1 / 3
    assert evaluator.evaluated == ["A"]
    assert updated_state.answers[0].gt_evaluation == Evaluation(score=1.0)
    assert updated_state.answers[1].gt_evaluation is None
    assert updated_state.answers[2].gt_evaluation is None
