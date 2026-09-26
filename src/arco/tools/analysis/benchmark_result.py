"""Benchmark artifact loading and tabular analysis models."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from arco.core import Answer, State
from arco.data import BenchmarkDataset


@dataclass(slots=True)
class BenchmarkResult:
    """Loaded artifacts from a completed benchmark run."""

    metadata: dict
    benchmark_dir: Path
    answer_analysis_df: pd.DataFrame
    run_analysis_df: pd.DataFrame
    _states: dict[str, dict[str, State]]
    benchmark_dataset: BenchmarkDataset | None = None

    @classmethod
    def load(cls, benchmark_dir: str) -> BenchmarkResult:
        bdir = Path(benchmark_dir)

        with open(bdir / "bench_metadata.json") as f:
            metadata = json.load(f)

        run_names = _load_run_names(bdir, metadata)
        runs_dir = bdir / "runs"

        # Check folder
        if not runs_dir.exists():
            raise ValueError("No runs directory available for this benchmark")
        run_csv_paths = sorted(runs_dir.glob("*.csv"))
        if not run_csv_paths:
            raise ValueError(f"No per-run CSV files found under {runs_dir}")

        # Build the answer_analysis_df and extract states
        answer_analysis_records: list[dict] = []
        states: dict[str, dict[str, State]] = {}
        for csv_path in run_csv_paths:
            # check csv
            file_run_name = csv_path.stem
            run_name = run_names.get(file_run_name, file_run_name)
            df = pd.read_csv(csv_path)
            required_columns = {"entry_id", "run_id", "state", "changes"}
            missing_columns = required_columns.difference(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Unsupported benchmark CSV {csv_path}; missing columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )

            # get run info and build analysis_df rows
            run_states: dict[str, State] = {}
            for _, row in df.iterrows():
                entry_id = int(row["entry_id"])  # pyright: ignore
                run_id = str(row["run_id"])
                # state
                state = State.from_dict(_load_json_dict(row.get("state"), default={}))
                # changes
                changes = _load_json_dict(row.get("changes"), default={})

                for trace_index, answer in enumerate(
                    state.answers
                ):  # one entry per answer
                    answer_analysis_records.append(
                        _answer_analysis_record(
                            entry_id=entry_id,
                            run_id=run_id,
                            trace_index=trace_index,
                            answer=answer,
                            changes=changes,
                            run_name=run_name,
                        )
                    )

                # store each state in a convenient dict
                run_states[str(row["run_id"])] = state

            if run_states:
                states[run_name] = run_states

        answer_analysis_df = pd.DataFrame(answer_analysis_records)
        analysis_dir = bdir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = analysis_dir / "benchmark.parquet"
        answer_analysis_df.to_parquet(parquet_path, index=False)
        answer_analysis_df = pd.read_parquet(parquet_path)

        # Build the run_analysis_df
        run_analysis_df = _build_run_analysis_df(answer_analysis_df)

        # Load the dataset
        dataset_path = metadata.get("dataset_path")
        dataset = None
        if dataset_path:
            dataset_file = Path(dataset_path)
            if not dataset_file.is_absolute():
                dataset_file = bdir / dataset_file
            if not dataset_file.exists():
                dataset_file = bdir / "dataset.json"
            if dataset_file.exists():
                dataset = BenchmarkDataset.from_json(str(dataset_file))

        return cls(
            metadata=metadata,
            benchmark_dir=bdir,
            answer_analysis_df=answer_analysis_df,
            run_analysis_df=run_analysis_df,
            benchmark_dataset=dataset,
            _states=states,
        )


def _load_json_dict(value: Any, *, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default
    return parsed if isinstance(parsed, dict) else default


def _load_run_names(benchmark_dir: Path, metadata: dict) -> dict[str, str]:
    """Map safe flat CSV stems back to the names configured for each run."""
    candidates = [metadata.get("benchmark_config_snapshot"), "benchmark_config.yaml"]
    for candidate in candidates:
        if not candidate:
            continue
        config_path = Path(candidate)
        if not config_path.is_absolute():
            config_path = benchmark_dir / config_path
        if not config_path.exists():
            config_path = benchmark_dir / Path(candidate).name
        if not config_path.exists():
            continue
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return {
            str(run.get("name", "")).replace(" ", "_"): str(run.get("name", ""))
            for run in config.get("runs", [])
        }
    return {}


def _answer_analysis_record(
    *,
    answer: Answer,
    run_name: str,
    run_id: str,
    trace_index: int,
    entry_id: int,
    changes: dict,
) -> dict:
    """Create a flat, typed row suitable for downstream analysis in Parquet."""
    profile = answer.profiling_data

    return {
        "run_name": run_name,  # benchmark run name
        "entry_id": entry_id,  # prompt_id
        "run_id": run_id,  # specific run id
        "trace_index": trace_index,  # answer number
        "agent": answer.agent_id,
        "message": answer.message,
        "score": answer.gt_evaluation.score if answer.gt_evaluation else None,
        "score_success": answer.gt_evaluation.success
        if answer.gt_evaluation
        else False,
        "best_of_n_score": answer.evaluation.score if answer.evaluation else None,
        "best_of_n_success": answer.evaluation.success if answer.evaluation else False,
        "perplexity": answer.perplexity if answer.perplexity is not None else None,
        "thinking": answer.thinking,
        "error": answer.error,
        "agent_output": json.dumps(
            answer.agent_output, ensure_ascii=False, sort_keys=True
        ),
        "agent_config": json.dumps(
            asdict(answer.agent_config), ensure_ascii=False, sort_keys=True
        ),
        "changes": json.dumps(changes, ensure_ascii=False, sort_keys=True),
        **profile.as_dict(),
    }


def _build_run_analysis_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the unified Parquet data into the a run-level aggregated view."""
    summary_rows = []
    if analysis_df.empty:
        return pd.DataFrame(summary_rows, columns=["name", "metrics_by_agent"])

    aggregated_metrics = {
        "evaluation_gt": "score",
        "perplexity": "perplexity",
        "total_time": "total_time",
        "llm_time": "llm_time",
    }
    for run_name, run_df in analysis_df.groupby("run_name", sort=False):
        metrics_by_agent: dict[str, dict[str, float]] = defaultdict(dict)

        for agent, agent_df in run_df.groupby("agent", sort=False):
            for summary_metric, column in aggregated_metrics.items():
                values = agent_df[column].dropna()
                if not values.empty:
                    metrics_by_agent[str(agent)][summary_metric] = float(values.mean())

        summary_rows.append(
            {
                "name": run_name,
                "metrics_by_agent": json.dumps(dict(metrics_by_agent)),
            }
        )
    return pd.DataFrame(summary_rows, columns=["name", "metrics_by_agent"])
