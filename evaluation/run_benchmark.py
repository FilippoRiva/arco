#!/usr/bin/env python3
"""Standalone benchmark script for evaluating the DataAgent against ground-truth datasets.

Usage:
    python evaluation/run_benchmark.py evaluation/benchmark_dataset.json --n 3
    python evaluation/run_benchmark.py evaluation/benchmark_dataset.json --n 1 --save-dir ./results
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.config import AgentConfig
from Agent.data_agent import SalesDataAgent
from Agent.utils import (
    compare_dataframes_iou,
    judge_analysis,
    judge_visualization,
    make_csv_evaluator_gt,
    make_csv_evaluator_no_gt,
    make_text_evaluator_gt,
    make_text_evaluator_no_gt,
    make_vis_evaluator_gt,
    make_vis_evaluator_no_gt,
)


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
    config_path: Optional[str] = None,
    n: int = 1,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    save_dir: str = "./evaluation/results",
) -> pd.DataFrame:
    """Run benchmark against a unified GT dataset.

    Args:
        dataset_path: Path to the benchmark JSON file.
        config_path: Optional path to run_config.yaml for base AgentConfig.
        n: Best-of-N per step.
        judge_model: Model for LLM-as-judge evaluations.
        judge_provider: Provider for judge model.
        save_dir: Directory to save results CSV.

    Returns:
        DataFrame with per-test-case scores.
    """
    entries = load_benchmark_dataset(dataset_path)
    print(f"Loaded {len(entries)} test cases from {dataset_path}")

    # Create base config
    if config_path:
        config, _run_params, _schema = AgentConfig.from_yaml(config_path)
    else:
        config = AgentConfig(model=judge_model, provider=judge_provider)

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
        if has_data:
            config.lookup_sales_data.n = n
            config.lookup_sales_data.gt_eval_fn = make_csv_evaluator_gt(
                ground_truth_csv_text=entry["gt_data"]
            )
            config.lookup_sales_data.batch_eval_fn = make_csv_evaluator_no_gt()
            config.lookup_sales_data.eval_fn = None
            config.lookup_sales_data.temp_min = 0.1
            config.lookup_sales_data.temp_max = 0.5

        if has_data and entry.get("gt_analysis"):
            config.analyzing_data.n = n
            config.analyzing_data.gt_eval_fn = make_text_evaluator_gt(
                ground_truth_text=entry["gt_analysis"],
                judge_model=judge_model,
                provider=judge_provider,
            )
            config.analyzing_data.eval_fn = make_text_evaluator_no_gt(
                judge_model=judge_model, provider=judge_provider
            )
            config.analyzing_data.temp_min = 0.1
            config.analyzing_data.temp_max = 0.7

        if has_vis:
            config.create_visualization.n = n
            config.create_visualization.gt_eval_fn = make_vis_evaluator_gt(
                ground_truth_config=entry["gt_chart_config"],
                ground_truth_code=entry.get("gt_chart_code", ""),
                explicit_requirements=entry.get("explicit_requirements"),
                judge_model=judge_model,
                provider=judge_provider,
            )
            config.create_visualization.eval_fn = make_vis_evaluator_no_gt(
                judge_model=judge_model, provider=judge_provider
            )
            config.create_visualization.temp_min = 0.1
            config.create_visualization.temp_max = 0.5

        # Run agent
        agent = SalesDataAgent(agent_config=config)
        output = agent.run(
            prompt,
            visualization_goal=vis_goal,
            no_vis=not has_vis,
            save_results=False,
        )
        result = output[0] if isinstance(output, tuple) else output

        # Compute final scores against GT
        row = {
            "test_case_id": idx,
            "prompt": prompt,
            "gen_sql": result.get("sql_query", ""),
        }

        # CSV/data IoU
        if has_data:
            gen_df = result.get("data_df")
            gt_df = pd.read_csv(pd.io.common.StringIO(entry["gt_data"]))
            row["csv_iou"] = compare_dataframes_iou(gen_df, gt_df) if gen_df is not None else 0.0
        else:
            row["csv_iou"] = None

        # Text analysis score
        if entry.get("gt_analysis") and result.get("answer"):
            score, _ = judge_analysis(
                prompt=prompt,
                sql_query=result.get("sql_query", ""),
                data=result.get("data", ""),
                analysis=result["answer"][0],
                judge_model=judge_model,
                provider=judge_provider,
            )
            row["text_score"] = score
        else:
            row["text_score"] = None

        # Visualization score
        if has_vis and result.get("chart_config"):
            answers = result.get("answer", [])
            chart_code = answers[-1] if len(answers) >= 2 else ""
            score, _ = judge_visualization(
                visualization_goal=vis_goal or prompt,
                generated_config=result["chart_config"],
                generated_code=chart_code,
                gt_config=entry["gt_chart_config"],
                gt_code=entry.get("gt_chart_code", ""),
                explicit_requirements=entry.get("explicit_requirements"),
                judge_model=judge_model,
                provider=judge_provider,
            )
            row["vis_score"] = score
        else:
            row["vis_score"] = None

        results.append(row)
        print(f"\nScores: csv_iou={row['csv_iou']}, text={row['text_score']}, vis={row['vis_score']}")

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
