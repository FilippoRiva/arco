#!/usr/bin/env python3
"""Aggregate bulk-run results into analysis-ready DataFrames.

Two output views
----------------
detail  — one row per (config_id × test_case_id); includes all config params
          and the three score columns (csv_iou, text_score, vis_score).

summary — one row per config_id; includes mean/std of each score column
          and all config param columns, sorted by csv_iou_mean descending.

Usage
-----
    # From Python
    from evaluation.aggregate_results import aggregate_bulk_results
    detail, summary = aggregate_bulk_results("evaluation/bulk_results/local_test")

    # From CLI
    python evaluation/aggregate_results.py evaluation/bulk_results/local_test
    python evaluation/aggregate_results.py evaluation/bulk_results/full_run --save
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_bulk_results(
    bulk_dir: str,
    *,
    save: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and aggregate all per-config benchmark results in *bulk_dir*.

    Parameters
    ----------
    bulk_dir : str
        Root directory produced by bulk_runner.py.
        Must contain ``configs_sampled.json`` and ``config_NNNN/`` subdirs.
    save : bool
        If True, write ``detail.csv`` and ``summary.csv`` inside *bulk_dir*.

    Returns
    -------
    detail : pd.DataFrame
        One row per (config_id × test_case_id).  All config params are merged
        in as extra columns.
    summary : pd.DataFrame
        One row per config_id.  Contains mean & std of each score metric.
        Sorted by csv_iou_mean descending (best data-retrieval accuracy first).
    """
    bulk_path = Path(bulk_dir)

    # --- Load config manifest ---
    manifest_path = bulk_path / "configs_sampled.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"configs_sampled.json not found in {bulk_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        config_records = json.load(f)
    configs_df = pd.DataFrame(config_records)

    # --- Collect per-config result CSVs ---
    result_frames = []
    for config_dir in sorted(bulk_path.glob("config_*")):
        results_csv = config_dir / "benchmark_results.csv"
        if not results_csv.exists():
            continue
        df = pd.read_csv(results_csv)

        # Infer config_id from directory name (config_0000 → 0)
        try:
            config_id = int(config_dir.name.split("_")[-1])
        except ValueError:
            continue

        df["config_id"] = config_id
        result_frames.append(df)

    if not result_frames:
        raise ValueError(f"No benchmark_results.csv files found under {bulk_dir}")

    all_results = pd.concat(result_frames, ignore_index=True)

    # --- Build detail view: merge config params into result rows ---
    detail = all_results.merge(configs_df, on="config_id", how="left")

    # Reorder: config_id first, then scores, then config params
    score_cols = [c for c in ("csv_iou", "text_score", "vis_score") if c in detail.columns]
    config_param_cols = [c for c in detail.columns if c not in ("config_id",) + tuple(score_cols)]
    ordered_cols = ["config_id"] + score_cols + config_param_cols
    detail = detail[[c for c in ordered_cols if c in detail.columns]]

    # --- Build summary view: one row per config ---
    agg_fns = {col: ["mean", "std"] for col in score_cols}
    summary_scores = (
        all_results.groupby("config_id")[score_cols]
        .agg(["mean", "std"])
    )
    summary_scores.columns = [f"{col}_{stat}" for col, stat in summary_scores.columns]
    summary_scores = summary_scores.reset_index()

    # Count test cases per config
    counts = all_results.groupby("config_id").size().reset_index(name="n_test_cases")
    summary_scores = summary_scores.merge(counts, on="config_id")

    # Merge config params
    summary = summary_scores.merge(configs_df, on="config_id", how="left")

    # Sort by csv_iou_mean descending (best retrieval accuracy first)
    if "csv_iou_mean" in summary.columns:
        summary = summary.sort_values("csv_iou_mean", ascending=False).reset_index(drop=True)

    # --- Optionally save ---
    if save:
        detail_path = bulk_path / "detail.csv"
        summary_path = bulk_path / "summary.csv"
        detail.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"Saved detail  → {detail_path}")
        print(f"Saved summary → {summary_path}")

    return detail, summary


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_summary(summary: pd.DataFrame, top_k: int = 10) -> None:
    """Print the top-k configs sorted by csv_iou_mean."""
    score_cols = [c for c in ("csv_iou_mean", "text_score_mean", "vis_score_mean") if c in summary.columns]
    step_n_cols = [c for c in summary.columns if c.endswith(".n")]

    print(f"\n{'rank':>4}  {'cfg':>4}  " + "  ".join(f"{c:>18}" for c in score_cols))
    print("-" * (10 + 20 * len(score_cols)))

    for rank, (_, row) in enumerate(summary.head(top_k).iterrows(), start=1):
        scores = "  ".join(
            f"{row[c]:>18.4f}" if pd.notna(row.get(c)) else f"{'N/A':>18}"
            for c in score_cols
        )
        n_str = " | ".join(
            f"{col.split('.')[0][:3]}.n={int(row[col])}"
            for col in step_n_cols
            if col in row and pd.notna(row[col])
        )
        print(f"{rank:>4}  {int(row['config_id']):>4}  {scores}   ({n_str})")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate bulk_runner results into detail and summary DataFrames"
    )
    parser.add_argument("bulk_dir", help="Root directory produced by bulk_runner.py")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write detail.csv and summary.csv inside bulk_dir",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top configs to print (default: 10)",
    )

    args = parser.parse_args()

    detail, summary = aggregate_bulk_results(args.bulk_dir, save=args.save)

    print(f"\nDetail  shape : {detail.shape}")
    print(f"Summary shape : {summary.shape}")
    print_summary(summary, top_k=args.top_k)


if __name__ == "__main__":
    main()
