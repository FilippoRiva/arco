import json
from types import SimpleNamespace

import pandas as pd

from run_agent import _write_execution_artifacts


def test_write_execution_artifacts_creates_expected_files(tmp_path):
    config_path = tmp_path / "run_config.yaml"
    config_path.write_text("run:\n  prompt: test\n", encoding="utf-8")

    result = {
        "run_id": "abc123",
        "answer": ["done"],
        "data_df": pd.DataFrame({"value": [1, 2]}),
    }
    agent_config = _make_agent_config()
    run_params = {
        "lookup_only": False,
        "no_vis": False,
        "reuse_from": None,
        "save_results": False,
    }

    artifact_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="Show me the sales in Nov 2021",
        result=result,
        agent_config=agent_config,
        run_params=run_params,
    )

    assert artifact_dir.is_dir()
    assert artifact_dir.name.endswith("_abc123_show_me_the_sales_in_nov_2021")
    assert (artifact_dir / "config_used.yaml").read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
    assert (artifact_dir / "effective_run_config.json").exists()

    metadata_text = (artifact_dir / "run_metadata.json").read_text(encoding="utf-8")
    assert '"run_id": "abc123"' in metadata_text
    metadata = json.loads(metadata_text)
    assert metadata["effective_run_params"]["save_results"] is False
    assert "lookup_only" not in metadata or "lookup_only" in metadata["effective_run_params"]
    # duplicate top-level run flags should no longer exist
    assert "save_execution_artifacts" not in metadata

    effective_config = (artifact_dir / "effective_run_config.json").read_text(encoding="utf-8")
    assert '"prompt": "Show me the sales in Nov 2021"' in effective_config
    assert '"provider": "openai"' in effective_config
    assert '"model": "gpt-4o-mini"' in effective_config

    result_json = (artifact_dir / "result.json").read_text(encoding="utf-8")
    assert '"__dataframe__": true' in result_json
    assert '"records"' in result_json


def test_write_execution_artifacts_uses_unique_directories_per_execution(tmp_path):
    config_path = tmp_path / "run_config.yaml"
    config_path.write_text("run:\n  prompt: test\n", encoding="utf-8")

    agent_config = _make_agent_config()
    run_params = {
        "lookup_only": True,
        "no_vis": True,
        "reuse_from": "cached-run",
        "save_results": True,
    }

    first_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="Repeated prompt",
        result={"run_id": "same-run"},
        agent_config=agent_config,
        run_params=run_params,
    )
    second_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="Repeated prompt",
        result={"run_id": "same-run"},
        agent_config=agent_config,
        run_params=run_params,
    )

    assert first_dir != second_dir
    assert first_dir.exists()
    assert second_dir.exists()
    assert (second_dir / "run_metadata.json").read_text(encoding="utf-8").find('"reuse_from": "cached-run"') != -1


def _make_agent_config(**kwargs):
    defaults = {"provider": "openai", "model": "gpt-4o-mini", "ollama_url": None, "to_dict": lambda: {}}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _base_fixtures(tmp_path):
    config_path = tmp_path / "run_config.yaml"
    config_path.write_text("run:\n  prompt: test\n", encoding="utf-8")
    agent_config = _make_agent_config()
    run_params = {"lookup_only": False, "no_vis": False, "reuse_from": None, "save_results": False}
    return config_path, agent_config, run_params


def test_metadata_contains_step_timings(tmp_path):
    config_path, agent_config, run_params = _base_fixtures(tmp_path)
    result = {
        "run_id": "t1",
        "_step_timings_sec": {"lookup_sales_data": 2.1, "analyzing_data": 4.5},
        "_total_run_time_sec": 6.6,
    }

    artifact_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="test prompt",
        result=result,
        agent_config=agent_config,
        run_params=run_params,
    )

    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["timing"]["total_run_time_sec"] == 6.6
    assert metadata["timing"]["step_timings_sec"]["lookup_sales_data"] == 2.1
    assert metadata["timing"]["step_timings_sec"]["analyzing_data"] == 4.5


def test_metadata_contains_energy(tmp_path):
    config_path, agent_config, run_params = _base_fixtures(tmp_path)
    energy = {
        "energy_consumed_kwh": 0.000123,
        "cpu_energy_kwh": 0.0001,
        "gpu_energy_kwh": 0.0,
        "ram_energy_kwh": 0.000023,
        "emissions_kg_co2": 0.0000456,
        "cpu_power_w": 15.0,
        "gpu_power_w": 0.0,
        "duration_sec": 6.6,
    }
    result = {"run_id": "t2", "_energy": energy}

    artifact_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="test prompt",
        result=result,
        agent_config=agent_config,
        run_params=run_params,
    )

    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["energy"]["energy_consumed_kwh"] == 0.000123
    assert metadata["energy"]["emissions_kg_co2"] == 0.0000456


def test_metadata_accuracy_present_when_gt_score_available(tmp_path):
    config_path, agent_config, run_params = _base_fixtures(tmp_path)
    result = {
        "run_id": "t3",
        "_gt_scores_per_step": {
            "lookup_sales_data": {"gt_score": 0.87, "all_gt_scores": [0.87, 0.72, 0.91]},
            "analyzing_data": {"gt_score": 0.65, "all_gt_scores": None},
        },
        "_step_eval_scores": {
            "lookup_sales_data": {"scores": [0.8, 0.9, 0.7], "best_idx": 1, "best_score": 0.9},
            "analyzing_data": {"scores": [0.75], "best_idx": 0, "best_score": 0.75},
        },
    }

    artifact_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="test prompt",
        result=result,
        agent_config=agent_config,
        run_params=run_params,
    )

    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["accuracy"]["type"] == "ground_truth"
    assert metadata["accuracy"]["ground_truth_scores"]["lookup_sales_data"]["gt_score"] == 0.87
    assert metadata["accuracy"]["ground_truth_scores"]["lookup_sales_data"]["all_gt_scores"] == [0.87, 0.72, 0.91]
    assert metadata["accuracy"]["ground_truth_scores"]["analyzing_data"]["gt_score"] == 0.65
    assert metadata["accuracy"]["step_eval_scores"]["analyzing_data"]["best_score"] == 0.75


def test_metadata_accuracy_uses_eval_scores_when_no_gt(tmp_path):
    config_path, agent_config, run_params = _base_fixtures(tmp_path)
    result = {
        "run_id": "t4",
        "_step_eval_scores": {
            "lookup_sales_data": {"scores": [0.8, 0.6, 0.9], "best_idx": 2, "best_score": 0.9},
            "analyzing_data": {"scores": [0.7, 0.85], "best_idx": 1, "best_score": 0.85},
        },
    }

    artifact_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="test prompt",
        result=result,
        agent_config=agent_config,
        run_params=run_params,
    )

    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["accuracy"]["type"] == "eval_scores"
    assert metadata["accuracy"]["step_eval_scores"]["lookup_sales_data"]["best_score"] == 0.9
    assert metadata["accuracy"]["step_eval_scores"]["analyzing_data"]["best_idx"] == 1


def test_metadata_accuracy_null_when_no_gt_and_no_eval_scores(tmp_path):
    config_path, agent_config, run_params = _base_fixtures(tmp_path)
    result = {"run_id": "t4b", "answer": ["some answer"]}

    artifact_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="test prompt",
        result=result,
        agent_config=agent_config,
        run_params=run_params,
    )

    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["accuracy"] is None


def test_metadata_timing_null_when_not_in_result(tmp_path):
    config_path, agent_config, run_params = _base_fixtures(tmp_path)
    result = {"run_id": "t5"}

    artifact_dir = _write_execution_artifacts(
        config_path=str(config_path),
        save_root=str(tmp_path / "output"),
        prompt="test prompt",
        result=result,
        agent_config=agent_config,
        run_params=run_params,
    )

    metadata = json.loads((artifact_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["timing"]["total_run_time_sec"] is None
    assert metadata["timing"]["step_timings_sec"] is None
