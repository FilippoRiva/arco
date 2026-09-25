import json

import pandas as pd
import pytest

import arco.tools.bench as bench_tools
from arco.core import Config, State
from arco.data import BenchmarkDataset, BenchmarkEntry, Trace


class FakeWorkflow:
    def __init__(self, *, fail_prompt: str | None = None):
        self.fail_prompt = fail_prompt
        self.prompts: list[str] = []

    def stream(self, config=None):
        self.prompts.append(config.prompt)
        if config.prompt == self.fail_prompt:
            raise RuntimeError("simulated interruption")
        state = State(
            prompt=config.prompt,
            run_id=config.run_id,
            agent_configs=config.agent_configs,
        )
        yield {"event": "completed", "state": state}

    def get_evaluators(self):
        return {}


def test_benchmark_from_config_resumes_only_missing_entries(tmp_path, monkeypatch):
    config = Config(workflow="fake", enable_storage=False)
    config_source = tmp_path / "source-benchmark.yaml"
    dataset_source = tmp_path / "source-dataset.json"
    config_source.write_text("global: {workflow: fake}\n")
    dataset_source.write_text("[]\n")
    dataset = BenchmarkDataset(
        [
            BenchmarkEntry(prompt="first", trace=Trace([]), id=10, difficulty=1),
            BenchmarkEntry(prompt="second", trace=Trace([]), id=20, difficulty=1),
        ]
    )
    run_config = {
        "name": "Baseline",
        "description": "resume test",
        "config": config,
        "changes": {},
    }
    workflow = FakeWorkflow(fail_prompt="second")

    monkeypatch.setattr(bench_tools.workflows, "load_workflows", lambda: ["fake"])
    monkeypatch.setattr(
        bench_tools.Config,
        "from_yaml",
        classmethod(lambda cls, path: config),
    )
    monkeypatch.setattr(
        bench_tools.Config,
        "generate_benchmark_configs",
        lambda self, path: [run_config],
    )
    monkeypatch.setattr(
        bench_tools.WorkflowFactory,
        "get",
        staticmethod(lambda config: workflow),
    )
    monkeypatch.setattr(
        bench_tools.BenchmarkDataset,
        "from_json",
        classmethod(lambda cls, path: dataset),
    )
    monkeypatch.setattr(bench_tools, "init_logging", lambda *args, **kwargs: None)

    def drain_with_interruption():
        for event in bench_tools.benchmark_from_config(
            config_path=str(config_source),
            dataset_path=str(dataset_source),
            id="resume-test",
            save_dir=str(tmp_path),
            logging_level=None,
        ):
            pass

    with pytest.raises(RuntimeError, match="simulated interruption"):
        drain_with_interruption()

    checkpoint = tmp_path / "resume-test" / "runs" / "Baseline.csv"
    first_checkpoint = pd.read_csv(checkpoint)
    assert first_checkpoint["entry_id"].tolist() == [10]
    assert workflow.prompts == ["first", "second"]

    resumed_workflow = FakeWorkflow()
    monkeypatch.setattr(
        bench_tools.WorkflowFactory,
        "get",
        staticmethod(lambda config: resumed_workflow),
    )
    events = list(
        bench_tools.benchmark_from_config(
            config_path=str(config_source),
            dataset_path=str(dataset_source),
            id="resume-test",
            save_dir=str(tmp_path),
            logging_level=None,
        )
    )

    assert resumed_workflow.prompts == ["second"]
    assert any(event["event"] == "benchmark_resume" for event in events)
    assert any(event["event"] == "benchmark_complete" for event in events)
    assert not (tmp_path / "resume-test" / "summary.csv").exists()
    assert (tmp_path / "resume-test" / "bench_metadata.json").exists()
    completed = pd.read_csv(checkpoint)
    assert completed["entry_id"].tolist() == [10, 20]
    assert completed["run_fingerprint"].nunique() == 1
    assert json.loads(completed.iloc[1]["state"])["prompt"] == "second"
    assert (tmp_path / "resume-test" / "benchmark_config.yaml").read_text() == (
        config_source.read_text()
    )
    assert (tmp_path / "resume-test" / "dataset.json").read_text() == (
        dataset_source.read_text()
    )

    cached_workflow = FakeWorkflow()
    monkeypatch.setattr(
        bench_tools.WorkflowFactory,
        "get",
        staticmethod(lambda config: cached_workflow),
    )
    cached_events = list(
        bench_tools.benchmark_from_config(
            config_path=str(config_source),
            dataset_path=str(dataset_source),
            id="resume-test",
            save_dir=str(tmp_path),
            logging_level=None,
        )
    )
    restored_event = next(
        event for event in cached_events if event["event"] == "benchmark_already_exists"
    )
    assert restored_event["cached_entries"] == 2
    assert cached_workflow.prompts == []
