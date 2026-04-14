#!/usr/bin/env python3
"""Standalone benchmark script for evaluating the DataAgent against ground-truth datasets.

Usage:
    python evaluation/run_benchmark.py evaluation/benchmark_dataset.json --n 3
    python evaluation/run_benchmark.py evaluation/benchmark_dataset.json --n 1 --save-dir ./results
"""

import argparse
import glob as _glob
import json
import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Agent.config import AgentConfig
from Agent.schema import ColumnSchema, DatabaseSchema, TableSchema
from Agent.utils import (
    make_csv_evaluator_gt,
    make_csv_evaluator_no_gt,
    make_text_evaluator_gt,
    make_text_evaluator_no_gt,
    make_vis_evaluator_gt,
    make_vis_evaluator_no_gt,
)
from run_agent import run_single  # non-interactive single-run entry point


def _load_schema(data_dir: Optional[str] = None) -> Optional[DatabaseSchema]:
    """Discover and load all *_schema.yaml files from data_dir.

    Mirrors the logic in AgentConfig.from_yaml so that run_benchmark has
    the same multi-table support as run_agent.
    """
    if data_dir is None:
        data_dir = os.path.join(_PROJECT_ROOT, "data")
    data_dir = os.path.abspath(data_dir)
    schema_files = sorted(_glob.glob(os.path.join(data_dir, "*_schema.yaml")))
    if not schema_files:
        return None
    tables = []
    for table_path in schema_files:
        with open(table_path, encoding="utf-8") as tf:
            t = yaml.safe_load(tf)
        columns = [
            ColumnSchema(
                name=c["name"],
                description=c.get("description", c["name"]),
                data_type=c.get("data_type", "VARCHAR"),
                example_values=c.get("example_values"),
                nullable=c.get("nullable", True),
            )
            for c in t.get("columns", [])
        ]
        file_path = t["file_path"]
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(table_path), file_path)
        tables.append(TableSchema(
            name=t["name"],
            description=t.get("description", t["name"]),
            file_path=file_path,
            columns=columns,
        ))
    return DatabaseSchema(tables=tables, compact_threshold=5)


def load_benchmark_dataset(path: str) -> List[Dict]:
    """Load and validate a unified GT dataset JSON."""
    with open(path) as f:
        entries = json.load(f)
    if not entries:
        raise ValueError(f"Empty dataset: {path}")
    first = entries[0]
    if "prompt" not in first:
        raise ValueError("Dataset entries must have a 'prompt' field")
    if "gt_data" not in first and "gt_chart_config" not in first:
        raise ValueError("Dataset entries must have 'gt_data' and/or 'gt_chart_config'")
    return entries


def run_benchmark(
    dataset_path: str,
    *,
    agent_config: Optional[AgentConfig] = None,
    config_path: Optional[str] = None,
    n: int = 1,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    save_dir: str = "./evaluation/results",
    data_dir: Optional[str] = None,
    save_execution_artifacts: bool = False,
    enable_codecarbon: bool = False,
    max_prompts: Optional[int] = None,
) -> pd.DataFrame:
    """Run benchmark against a unified GT dataset.

    Args:
        dataset_path: Path to the benchmark JSON file.
        agent_config: Pre-built AgentConfig to use directly. When provided,
            sampling parameters (n, temp, top_p) are preserved as-is; only
            eval functions are attached from the benchmark GT data.
            Mutually exclusive with config_path / n.
        config_path: Optional path to run_config.yaml for base AgentConfig.
            Ignored when agent_config is provided.
        n: Best-of-N per step. Ignored when agent_config is provided.
        judge_model: Model for LLM-as-judge evaluations.
            Defaults to agent_config.model when agent_config is provided.
        judge_provider: Provider for judge model.
            Defaults to agent_config.provider when agent_config is provided.
        save_dir: Directory to save results CSV.

    Returns:
        DataFrame with per-test-case scores.
    """
    entries = load_benchmark_dataset(dataset_path)
    if max_prompts is not None:
        entries = entries[:max_prompts]
    print(f"Loaded {len(entries)} test cases from {dataset_path}")

    # Load schema once (covers all tables: sales, stores, products, ...)
    # Same logic as AgentConfig.from_yaml — without this, SalesDataAgent falls
    # back to single-table legacy mode and JOIN queries on stores/products fail.
    schema = _load_schema(data_dir)
    if schema:
        print(f"Loaded schema: {[t.name for t in schema.tables]}")

    # Determine base config and judge identity
    _external_config = agent_config is not None
    if _external_config:
        config = agent_config
        judge_model = config.model
        judge_provider = config.provider
    elif config_path:
        config, _run_params, _schema = AgentConfig.from_yaml(config_path)
    else:
        config = AgentConfig(
            model=judge_model,
            provider=judge_provider,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )

    results = []

    for idx, entry in enumerate(entries):
        prompt = entry["prompt"]
        vis_goal = entry.get("visualization_goal")
        has_vis = entry.get("gt_chart_config") is not None
        has_data = entry.get("gt_data") is not None

        print(f"\n{'='*60}")
        print(f"TEST CASE {idx + 1}/{len(entries)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Has data GT: {has_data} | Has vis GT: {has_vis}")

        # Configure step-level eval functions for this entry.
        # GT eval functions are used for tracking/logging only (gt_eval_fn).
        # Non-GT eval functions are used for best-of-n selection (eval_fn / batch_eval_fn).
        # When _external_config=True, sampling params (n, temp, top_p) are NOT overridden.
        if has_data:
            if not _external_config:
                config.lookup_sales_data.n = n
                config.lookup_sales_data.temp_min = 0.1
                config.lookup_sales_data.temp_max = 0.5
            config.lookup_sales_data.gt_eval_fn = make_csv_evaluator_gt(
                ground_truth_csv_text=entry["gt_data"]
            )
            config.lookup_sales_data.batch_eval_fn = make_csv_evaluator_no_gt()
            config.lookup_sales_data.eval_fn = None
            # Force column names to GT columns during standardize_candidate_columns
            # (same as AgentConfig.from_yaml lines 417/423 — missing here was the root cause of csv_iou=0)
            _gt_df = pd.read_csv(pd.io.common.StringIO(entry["gt_data"]))
            config.lookup_sales_data.gt_columns = [c.lower() for c in _gt_df.columns]

        if has_data and entry.get("gt_analysis"):
            if not _external_config:
                config.analyzing_data.n = n
                config.analyzing_data.temp_min = 0.1
                config.analyzing_data.temp_max = 0.7
            config.analyzing_data.gt_eval_fn = make_text_evaluator_gt(
                ground_truth_text=entry["gt_analysis"],
                judge_model=judge_model,
                provider=judge_provider,
            )
            config.analyzing_data.eval_fn = make_text_evaluator_no_gt(
                judge_model=judge_model,
                provider=judge_provider,
                ollama_url=config.ollama_url,
                openai_api_key=config.openai_api_key,
            )

        if has_vis:
            if not _external_config:
                config.create_visualization.n = n
                config.create_visualization.temp_min = 0.1
                config.create_visualization.temp_max = 0.5
            config.create_visualization.gt_eval_fn = make_vis_evaluator_gt(
                ground_truth_config=entry["gt_chart_config"],
                ground_truth_code=entry.get("gt_chart_code", ""),
                explicit_requirements=entry.get("explicit_requirements"),
                judge_model=judge_model,
                provider=judge_provider,
            )
            config.create_visualization.eval_fn = make_vis_evaluator_no_gt(
                judge_model=judge_model,
                provider=judge_provider,
                ollama_url=config.ollama_url,
                openai_api_key=config.openai_api_key,
            )

        # Run agent via run_single (same path as run_agent.py — no logic duplication)
        result = run_single(
            config,
            prompt,
            schema,
            visualization_goal=vis_goal,
            no_vis=not has_vis,
            save_dir=save_dir,
            save_results=False,
            save_execution_artifacts=save_execution_artifacts,
            enable_codecarbon=enable_codecarbon,
        )

        # --- Extract scores from result (same path as run_agent.py) ---
        # gt_eval_fn was configured above and called inside agent.run(); scores
        # are already normalised (column name forcing, LLM judge, etc.).
        gt_scores = result.get("_gt_scores_per_step", {})
        eval_scores = result.get("_step_eval_scores", {})

        # --- Per-prompt timing (bug fix: was the total run time repeated on every row) ---
        step_timings = result.get("_step_timings_sec", {})
        total_time = result.get("_total_run_time_sec")

        # --- Energy (populated only when enable_codecarbon=True) ---
        energy = result.get("_energy") or {}

        row = {
            "test_case_id": idx,
            "prompt": prompt,
            "gen_sql": " ".join((result.get("sql_query", "") or "").split()),
            # GT scores — same source as run_metadata.json accuracy.ground_truth_scores
            "csv_iou":    gt_scores.get("lookup_sales_data", {}).get("gt_score") if has_data else None,
            "text_score": gt_scores.get("analyzing_data", {}).get("gt_score") if entry.get("gt_analysis") else None,
            "vis_score":  gt_scores.get("create_visualization", {}).get("gt_score") if has_vis else None,
            # No-GT quality scores (BoN selector) — same source as run_metadata.json accuracy.step_eval_scores
            "csv_eval_score":  eval_scores.get("lookup_sales_data", {}).get("best_score") if has_data else None,
            "text_eval_score": eval_scores.get("analyzing_data", {}).get("best_score") if entry.get("gt_analysis") else None,
            "vis_eval_score":  eval_scores.get("create_visualization", {}).get("best_score") if has_vis else None,
            # Per-prompt timing
            "elapsed_sec":        round(total_time, 2) if total_time is not None else None,
            "lookup_time_sec":    round(step_timings.get("lookup_sales_data", 0), 2),
            "analyzing_time_sec": round(step_timings.get("analyzing_data", 0), 2),
            "vis_time_sec":       round(step_timings.get("create_visualization", 0), 2),
            # Energy (None when CodeCarbon disabled or unavailable)
            "energy_consumed_kwh": energy.get("energy_consumed_kwh"),
            "cpu_energy_kwh":      energy.get("cpu_energy_kwh"),
            "gpu_energy_kwh":      energy.get("gpu_energy_kwh"),
            "ram_energy_kwh":      energy.get("ram_energy_kwh"),
            "emissions_kg_co2":    energy.get("emissions_kg_co2"),
        }

        results.append(row)
        print(
            f"\nScores: csv_iou={row['csv_iou']}, text={row['text_score']}, vis={row['vis_score']}"
            f" | elapsed={row['elapsed_sec']}s"
        )

    # Build results DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for col in ["csv_iou", "text_score", "vis_score"]:
        valid = df[col].dropna()
        if not valid.empty:
            print(f"  {col}: mean={valid.mean():.3f}, min={valid.min():.3f}, max={valid.max():.3f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "benchmark_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DataAgent benchmark against GT dataset")
    parser.add_argument("dataset", help="Path to benchmark dataset JSON")
    parser.add_argument("--n", type=int, default=1, help="Best-of-N per step (default: 1)")
    parser.add_argument("--config", default=None, help="Path to run_config.yaml")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Judge model (default: gpt-4o-mini)")
    parser.add_argument("--judge-provider", default="openai", help="Judge provider (default: openai)")
    parser.add_argument("--save-dir", default="./evaluation/results", help="Output directory")

    args = parser.parse_args()

    run_benchmark(
        args.dataset,
        config_path=args.config,
        n=args.n,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        save_dir=args.save_dir,
    )
