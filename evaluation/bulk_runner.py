#!/usr/bin/env python3
"""Bulk runner: samples random AgentConfig instances from a search space and
evaluates each on the full benchmark dataset.

Think time between consecutive config runs is drawn from an Exponential(mean)
distribution to avoid hammering the API.

Usage — local test (3 configs):
    python evaluation/bulk_runner.py \\
        evaluation/benchmark_dataset.json \\
        evaluation/search_space.yaml \\
        --n-configs 3 \\
        --save-dir evaluation/bulk_results/local_test

Usage — full run (50 configs):
    python evaluation/bulk_runner.py \\
        evaluation/benchmark_dataset.json \\
        evaluation/search_space.yaml \\
        --n-configs 50 \\
        --think-time 5.0 \\
        --save-dir evaluation/bulk_results/full_run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.config import AgentConfig  # noqa: E402
from evaluation.run_benchmark import run_benchmark  # noqa: E402
from evaluation.search_space import SearchSpace  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_config_summary(config_id: int, cfg: AgentConfig, n_total: int) -> None:
    """Print a compact one-line summary for each sampled config."""
    parts = [f"CONFIG {config_id + 1}/{n_total}  model={cfg.model}"]
    for step in ("lookup_sales_data", "analyzing_data", "create_visualization"):
        sc = cfg.get_step_config(step)
        abbr = {"lookup_sales_data": "lsd", "analyzing_data": "ana", "create_visualization": "cvi"}[step]
        if sc.bon_param == "temperature":
            param_str = f"T[{sc.temp_min:.2f},{sc.temp_max:.2f}]"
        else:
            param_str = f"P[{sc.top_p_min:.2f},{sc.top_p_max:.2f}]"
        parts.append(f"{abbr}:n={sc.n},{param_str}")
    print("  " + " | ".join(parts))


def _exponential_think_time(mean: float, rng: np.random.RandomState) -> float:
    """Draw a think-time sample from Exponential(mean).  Returns seconds."""
    return float(rng.exponential(scale=mean))


# ---------------------------------------------------------------------------
# Core bulk runner
# ---------------------------------------------------------------------------

def run_bulk_benchmark(
    dataset_path: str,
    search_space_path: str,
    *,
    n_configs: int = 3,
    seed: int = 42,
    think_time_mean: float = 5.0,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    openai_api_key: Optional[str] = None,
    save_dir: str = "./evaluation/bulk_results",
    resume: bool = False,
) -> pd.DataFrame:
    """Run N random configs on the benchmark and aggregate results.

    Parameters
    ----------
    dataset_path : str
        Path to the benchmark JSON file (e.g. evaluation/benchmark_dataset.json).
    search_space_path : str
        Path to the search space YAML (e.g. evaluation/search_space.yaml).
    n_configs : int
        Number of random configs to sample and evaluate.
    seed : int
        Master seed for config sampling *and* think-time draws.
    think_time_mean : float
        Mean of the Exponential distribution for inter-run sleep (seconds).
        Set to 0 to disable think time.
    model : str
        LLM model name passed to every sampled AgentConfig.
    provider : str
        LLM provider ("openai" or "ollama").
    openai_api_key : str, optional
        OpenAI API key; defaults to OPENAI_API_KEY env var.
    save_dir : str
        Root directory where per-config subdirs and summary CSV are saved.
    resume : bool
        If True, skip configs whose output directory already exists.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per config.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # --- Build base config (model / provider only; steps overridden by space) ---
    base_config = AgentConfig(
        model=model,
        provider=provider,
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
    )

    # --- Sample all configs upfront so the full plan is visible & reproducible ---
    space = SearchSpace(search_space_path)
    configs: List[AgentConfig] = space.sample(
        n_configs=n_configs, seed=seed, base_config=base_config
    )

    # Persist sampled configs manifest (written before any run starts)
    records = space.configs_to_records(configs)
    manifest_path = save_path / "configs_sampled.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Sampled {n_configs} configs → {manifest_path}")
    print()

    # RNG for think-time draws (separate from config-sampling RNG)
    think_rng = np.random.RandomState(seed + 1)

    all_results: List[pd.DataFrame] = []

    for config_idx, agent_config in enumerate(configs):
        config_dir = save_path / f"config_{config_idx:04d}"

        # Resume: skip if already run
        done_marker = config_dir / "benchmark_results.csv"
        if resume and done_marker.exists():
            print(f"[config {config_idx:04d}] Skipping (already done).")
            df_existing = pd.read_csv(done_marker)
            df_existing["config_id"] = config_idx
            all_results.append(df_existing)
            continue

        config_dir.mkdir(parents=True, exist_ok=True)

        # Print header
        print(f"\n{'='*70}")
        _print_config_summary(config_idx, agent_config, n_configs)
        print(f"{'='*70}")

        # Save this config's parameters
        with open(config_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(records[config_idx], f, indent=2)

        # --- Run benchmark ---
        t_start = time.perf_counter()
        df_config = run_benchmark(
            dataset_path,
            agent_config=agent_config,
            save_dir=str(config_dir),
        )
        elapsed = time.perf_counter() - t_start

        # Tag with config metadata
        df_config["config_id"] = config_idx
        df_config["elapsed_sec"] = round(elapsed, 2)

        df_config.to_csv(config_dir / "benchmark_results.csv", index=False)
        all_results.append(df_config)

        print(f"\n[config {config_idx:04d}] Done in {elapsed:.1f}s")

        # --- Think time (not after last config) ---
        if config_idx < n_configs - 1 and think_time_mean > 0:
            t_sleep = _exponential_think_time(think_time_mean, think_rng)
            print(f"Think time: sleeping {t_sleep:.1f}s before next config…")
            time.sleep(t_sleep)

    # --- Build & save summary ---
    if not all_results:
        print("No results to aggregate.")
        return pd.DataFrame()

    summary = _build_summary(all_results, records)
    summary_path = save_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print("BULK RUN COMPLETE")
    print(f"{'='*70}")
    print(f"Configs run : {len(all_results)}")
    print(f"Results     : {save_dir}")
    _print_summary_table(summary)

    return summary


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def _build_summary(
    all_results: List[pd.DataFrame],
    config_records: List[dict],
) -> pd.DataFrame:
    """Aggregate per-config DataFrames into a single summary DataFrame.

    Each row = one config.  Columns include mean/std of csv_iou, text_score,
    vis_score across all test cases, plus all config parameters.
    """
    rows = []
    for cfg_rec in config_records:
        config_id = cfg_rec["config_id"]
        # Find matching result frame
        df = next((r for r in all_results if int(r["config_id"].iloc[0]) == config_id), None)
        if df is None:
            continue

        row = dict(cfg_rec)  # all config params

        for metric in ("csv_iou", "text_score", "vis_score"):
            if metric in df.columns:
                valid = df[metric].dropna()
                row[f"{metric}_mean"] = round(valid.mean(), 4) if not valid.empty else None
                row[f"{metric}_std"] = round(valid.std(), 4) if len(valid) > 1 else 0.0
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None

        if "elapsed_sec" in df.columns:
            row["elapsed_sec"] = round(df["elapsed_sec"].iloc[0], 2)

        rows.append(row)

    return pd.DataFrame(rows)


def _print_summary_table(summary: pd.DataFrame) -> None:
    """Print a compact table of mean scores per config."""
    metric_cols = [c for c in ("csv_iou_mean", "text_score_mean", "vis_score_mean") if c in summary.columns]
    if not metric_cols:
        return
    print(f"\n{'cfg':>4}  " + "  ".join(f"{c:>18}" for c in metric_cols))
    print("-" * (6 + 20 * len(metric_cols)))
    for _, row in summary.iterrows():
        vals = "  ".join(
            f"{row[c]:>18.4f}" if row[c] is not None else f"{'N/A':>18}"
            for c in metric_cols
        )
        print(f"{int(row['config_id']):>4}  {vals}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk runner: random search over AgentConfig hyperparameters"
    )
    parser.add_argument("dataset", help="Path to benchmark dataset JSON")
    parser.add_argument("search_space", help="Path to search_space.yaml")
    parser.add_argument(
        "--n-configs",
        type=int,
        default=3,
        help="Number of random configs to evaluate (default: 3 for local test, 50 for full)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument(
        "--think-time",
        type=float,
        default=5.0,
        help="Mean of Exponential think time between runs in seconds (default: 5.0; 0 to disable)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--save-dir", default="./evaluation/bulk_results", help="Output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs whose output directory already exists",
    )

    args = parser.parse_args()

    run_bulk_benchmark(
        args.dataset,
        args.search_space,
        n_configs=args.n_configs,
        seed=args.seed,
        think_time_mean=args.think_time,
        model=args.model,
        provider=args.provider,
        save_dir=args.save_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
