from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from arco.cli.viz.panels import _verbose_token_summary
from arco.core import AgentType, Answer, Config, ProfilingData, State
from arco.core.agent import _ACTIVE_LLM_ACCUMULATOR, Agent
from arco.core.agent_config import AgentConfig
from arco.core.answer import AnswerDraft
from arco.core.llm_tools import LLMAnswer
from arco.core.tracking import LLMCallAccumulator, TokenUsage


def _usage(input_tokens: int, output_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {
            "audio": 10,
            "cache_creation": 3,
            "cache_read": 5,
        },
        "output_token_details": {"audio": 10, "reasoning": 7},
    }


def test_llm_answer_extracts_standard_usage_metadata():
    usage = _usage(input_tokens=30, output_tokens=12)
    answer = LLMAnswer(AIMessage(content="answer", usage_metadata=usage))

    assert answer.input_token_count == 30
    assert answer.output_token_count == 12
    assert answer.total_token_count == 42
    assert answer.cache_creation_token_count == 3
    assert answer.cache_read_token_count == 5
    assert answer.reasoning_token_count == 7


def test_call_accumulator_sums_usage_across_llm_calls():
    accumulator = LLMCallAccumulator("token-test")
    for input_tokens, output_tokens in ((30, 12), (10, 4)):
        message = AIMessage(
            content="answer",
            additional_kwargs={"reasoning_content": "considered the evidence"},
            response_metadata={
                "logprobs": {"content": [{"token": "answer", "logprob": -0.2}]},
                "model_name": "test-model",
            },
            usage_metadata=_usage(input_tokens, output_tokens),
        )
        result = LLMResult(generations=[[ChatGeneration(message=message)]])
        accumulator.on_llm_end(result, run_id=f"call-{input_tokens}")

    assert accumulator.token_usage == TokenUsage(
        input_tokens=40,
        output_tokens=16,
        total_tokens=56,
        cache_creation_tokens=6,
        cache_read_tokens=10,
        reasoning_tokens=14,
    )
    assert accumulator.last_logprobs == [("answer", -0.2)]
    assert accumulator.last_reasoning == "considered the evidence"


def test_agent_answer_gets_reasoning_and_logprobs_from_framework_context():
    class MetadataAgent(Agent):
        def core(self, state, llm) -> AnswerDraft:
            return AnswerDraft(message="answer")

    config = Config(workflow="metadata-test")
    state = State(
        prompt="Generate an answer.",
        run_id="metadata-test",
        agent_configs=config.agent_configs,
    )
    accumulator = LLMCallAccumulator("metadata-test")
    accumulator.last_logprobs = [("answer", -0.2)]
    accumulator.last_reasoning = "considered the evidence"
    context_token = _ACTIVE_LLM_ACCUMULATOR.set(accumulator)
    try:
        updated_state = MetadataAgent()._run_core_and_populate_answer(
            state, cast(Any, object())
        )
    finally:
        _ACTIVE_LLM_ACCUMULATOR.reset(context_token)

    answer = updated_state.get_last_answer(AgentType("MetadataAgent"))
    assert answer is not None
    assert answer.logprobs == [("answer", -0.2)]
    assert answer.thinking == "considered the evidence"


def test_verbose_token_summary_shows_usage_without_logprobs():
    answer = Answer(
        agent_id=AgentType("TestAgent"),
        message="Answer",
        agent_config=AgentConfig(),
        logprobs=[],
        input_token_count=350,
        output_token_count=240,
        total_token_count=590,
        cache_creation_token_count=200,
        cache_read_token_count=100,
        reasoning_token_count=200,
        profiling_data=ProfilingData(llm_time=2.0),
    )

    summary = str(_verbose_token_summary(answer))
    assert "Input 350 tokens" in summary
    assert "Output 240 tokens" in summary
    assert "120.0 output tokens/s" in summary
    assert "Total 590 tokens" in summary
    assert "Cache read 100 tokens" in summary
    assert "Cache creation 200 tokens" in summary
    assert "Reasoning 200 tokens" in summary
