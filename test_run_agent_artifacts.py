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
    agent_config = SimpleNamespace(provider="openai", model="gpt-4o-mini")
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

    metadata = (artifact_dir / "run_metadata.json").read_text(encoding="utf-8")
    assert '"run_id": "abc123"' in metadata
    assert '"save_results": false' in metadata

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

    agent_config = SimpleNamespace(provider="openai", model="gpt-4o-mini")
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
