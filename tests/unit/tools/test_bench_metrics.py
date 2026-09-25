import json

import pytest

from arco.core import (
    AgentConfig,
    AgentType,
    Answer,
    Config,
    Evaluation,
    State,
)
from arco.data import BenchmarkDataset, BenchmarkEntry, Trace
from arco.tools.bench import (
    _append_checkpoint_row,
    _load_checkpoint,
    _prepare_input_snapshot,
    _run_fingerprint,
    benchmark,
)


def test_input_snapshots_are_reused_and_reject_changed_sources(tmp_path):
    source = tmp_path / "source.yaml"
    snapshot = tmp_path / "experiment" / "benchmark_config.yaml"
    source.write_text("first: true\n")

    path, copied = _prepare_input_snapshot(str(source), snapshot)
    assert path == snapshot
    assert copied is True
    assert snapshot.read_text() == "first: true\n"

    _, copied_again = _prepare_input_snapshot(str(source), snapshot)
    assert copied_again is False
    source.write_text("first: false\n")
    with pytest.raises(ValueError, match="differs from the existing snapshot"):
        _prepare_input_snapshot(str(source), snapshot)


def test_checkpoint_restores_completed_entries_and_recovers_partial_trailing_row(
    tmp_path,
):
    checkpoint = tmp_path / "run.csv"
    row = {
        "entry_id": 3,
        "run_id": "run3",
        "run_fingerprint": "abc123",
        "changes": json.dumps({}),
        "state": json.dumps({"prompt": "p"}),
    }
    _append_checkpoint_row(checkpoint, row)
    with checkpoint.open("ab") as file:
        file.write(b'4,run4,abc123,"partial')

    restored, recovered = _load_checkpoint(
        checkpoint, expected_entry_ids={3, 4}, fingerprint="abc123"
    )

    assert recovered is True
    assert set(restored) == {3}
    assert restored[3]["run_id"] == "run3"


def test_checkpoint_rejects_rows_from_different_inputs(tmp_path):
    checkpoint = tmp_path / "run.csv"
    _append_checkpoint_row(
        checkpoint,
        {
            "entry_id": 1,
            "run_id": "run1",
            "run_fingerprint": "old",
            "changes": json.dumps({}),
            "state": json.dumps({}),
        },
    )

    with pytest.raises(ValueError, match="different benchmark inputs"):
        _load_checkpoint(checkpoint, expected_entry_ids={1}, fingerprint="new")


def test_run_fingerprint_changes_when_effective_config_or_dataset_changes():
    config = Config(workflow="test", default_model="model-a")
    dataset_a = BenchmarkDataset(
        [BenchmarkEntry(prompt="one", trace=Trace([]), id=1, difficulty=1)]
    )
    dataset_b = BenchmarkDataset(
        [BenchmarkEntry(prompt="two", trace=Trace([]), id=1, difficulty=1)]
    )

    original = _run_fingerprint(config, {}, dataset_a)
    assert original == _run_fingerprint(config, {}, dataset_a)
    assert original != _run_fingerprint(config.set(default_model="model-b"), {}, dataset_a)
    assert original != _run_fingerprint(config, {}, dataset_b)


def test_benchmark_skips_cached_entries_and_keeps_original_case_position():
    entries = [
        BenchmarkEntry(prompt="first", trace=Trace([]), id=10, difficulty=1),
        BenchmarkEntry(prompt="second", trace=Trace([]), id=20, difficulty=1),
    ]
    dataset = BenchmarkDataset(entries)
    state = State(prompt="second", run_id="run20", agent_configs={})

    class FakeWorkflow:
        def __init__(self):
            self.prompts: list[str] = []

        def stream(self, config=None):
            self.prompts.append(config.prompt)
            yield {"event": "completed", "state": state}

        def get_evaluators(self):
            return {}

    workflow = FakeWorkflow()
    events = list(
        benchmark(
            workflow=workflow,
            name="run",
            config=Config(workflow="test", prompt="base"),
            changes={},
            benchmark_dataset=dataset,
            completed_entry_ids={10},
            run_fingerprint="hash",
        )
    )

    starts = [event for event in events if event["event"] == "test_case_start"]
    rows = [event["row"] for event in events if event["event"] == "entry_result"]
    assert workflow.prompts == ["second"]
    assert starts[0]["iteration"] == 2
    assert starts[0]["entry_id"] == 20
    assert [row["entry_id"] for row in rows] == [20]


def test_state_serialization_preserves_zero_perplexity_and_score():
    answer = Answer(
        agent_id=AgentType("Analyzer"),
        message="",
        agent_config=AgentConfig(),
        logprobs=[],
        evaluation=Evaluation(score=0.0),
        gt_evaluation=Evaluation(score=0.0),
        perplexity=0.0,
    )

    serialized = json.dumps(answer.to_dict())
    restored = Answer.from_dict(json.loads(serialized))

    assert restored.perplexity == 0.0
    assert restored.gt_evaluation == Evaluation(score=0.0)


def test_per_entry_output_contains_complete_state_without_redundant_trace():
    answer = Answer(
        agent_id=AgentType("Agent"),
        message="done",
        agent_config=AgentConfig(),
        logprobs=[],
        agent_output={"result": "full output"},
        perplexity=0.0,
    )
    final_state = State(
        prompt="question",
        run_id="run0",
        agent_configs={},
        answers=(answer,),
    )

    class FakeWorkflow:
        def stream(self, config=None):
            yield {"event": "completed", "state": final_state}

        def get_evaluators(self):
            return {}

    dataset = BenchmarkDataset(
        [BenchmarkEntry(prompt="question", trace=Trace([]), id=0, difficulty=1)]
    )
    result_event = next(
        event
        for event in benchmark(
            workflow=FakeWorkflow(),
            name="run",
            config=Config(workflow="test", prompt="question"),
            changes={"Analyzer": {"max_tokens": 4000}},
            benchmark_dataset=dataset,
            run_fingerprint="fingerprint",
        )
        if event["event"] == "entry_result"
    )
    row = result_event["row"]
    state_data = json.loads(row["state"])
    assert set(row) == {
        "entry_id",
        "run_id",
        "run_fingerprint",
        "changes",
        "state",
    }
    assert json.loads(row["changes"]) == {"Analyzer": {"max_tokens": 4000}}
    assert row["run_fingerprint"] == "fingerprint"
    assert state_data["answers"][0]["agent_output"] == {"result": "full output"}
    assert state_data["answers"][0]["perplexity"] == 0.0
    restored_state = State.from_dict(state_data)
    assert restored_state.answers[0].agent_output["result"] == "full output"
    assert restored_state.answers[0].perplexity == 0.0


