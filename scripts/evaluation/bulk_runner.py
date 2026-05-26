#!/usr/bin/env python3
"""Bulk runner: 3-phase ablation study over AgentConfig hyperparameters.

In each phase only ONE step's hyperparameters are varied across N random
configs; the other two steps are kept at their default (n=1, fixed temp).
This implements the 50+50+50 protocol requested by the professor.

Think time between consecutive config runs is drawn from Exponential(mean)
to avoid hammering the API.

Usage — validation (1 config per phase, no think time):
    python evaluation/bulk_runner.py \\
        evaluation/benchmark_dataset.json \\
        evaluation/search_space.yaml \\
        --n-configs 1 --think-time 0 \\
        --save-dir runs/bulk_results/validation

Usage — full 50+50+50 run:
    python evaluation/bulk_runner.py \\
        evaluation/benchmark_dataset.json \\
        evaluation/search_space.yaml \\
        --n-configs 50 --think-time 5.0 \\
        --save-dir runs/bulk_results/full_run

Usage — resume/run only one specific phase:
    python evaluation/bulk_runner.py ... --vary-step lookup_sales_data --resume
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

from core.config import ArcoConfig  # noqa: E402
from evaluators import aggregate_bulk_results  # noqa: E402
from evaluators import run_benchmark  # noqa: E402
from evaluators import SearchSpace  # noqa: E402


# The three ablation phases, in execution order
_VARY_STEPS = ["lookup_sales_data", "analyzing_data", "create_visualization"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_config_summary(config_id: int, cfg: ArcoConfig, n_total: int) -> None:
    """Print a detailed multi-line summary for each sampled config."""
    label = {
        "lookup_sales_data": "lookup_sales_data ",
        "analyzing_data": "analyzing_data    ",
        "create_visualization": "create_vis        ",
    }
    is_openai = cfg.provider in ("openai",)
    print(f"  CONFIG {config_id + 1}/{n_total}  model={cfg.model}  provider={cfg.provider}")
    for step in _VARY_STEPS:
        sc = cfg.get_agent_config(step)
        if sc.bon_parameter == "temperature":
            bon_str = f"bon=temperature  T[{sc.temp_min:.2f}→{sc.temp_max:.2f}]  top_p={sc.top_p_min:.2f}(fixed)"
        elif sc.bon_parameter == "top_p":
            bon_str = f"bon=top_p        temp={sc.temp_min:.2f}(fixed)  P[{sc.top_p_min:.2f}→{sc.top_p_max:.2f}]"
        else:  # top_k
            bon_str = f"bon=top_k        temp={sc.temp_min:.2f}(fixed)  top_p={sc.top_p_min:.2f}(fixed)  K[{sc.top_k_min}→{sc.top_k_max}]"
        line = f"    {label[step]}: n={sc.n}  cot_n={sc.cot_n}  {bon_str}  max_tokens={sc.max_tokens}"
        if not is_openai:
            top_k_str = f"top_k={sc.top_k_min}" if sc.bon_parameter != "top_k" else "(BoN axis)"
            line += f"  | {top_k_str}  beams={sc.num_beams}  no_repeat_ngram={sc.no_repeat_ngram_size}"
        print(line)


def _exponential_think_time(mean: float, rng: np.random.RandomState) -> float:
    """Draw a think-time sample from Exponential(mean).  Returns seconds."""
    return float(rng.exponential(scale=mean))


def _print_summary_table(summary: pd.DataFrame) -> None:
    """Print a compact table of mean scores per config."""
    metric_cols = [c for c in ("csv_iou_mean", "text_score_mean", "vis_score_mean") if c in summary.columns]
    if not metric_cols:
        return
    print(f"\n{'cfg':>4}  " + "  ".join(f"{c:>18}" for c in metric_cols))
    print("-" * (6 + 20 * len(metric_cols)))
    for _, row in summary.iterrows():
        vals = "  ".join(
            f"{row[c]:>18.4f}" if pd.notna(row.get(c)) else f"{'N/A':>18}"
            for c in metric_cols
        )
        print(f"{int(row['config_id']):>4}  {vals}")


# ---------------------------------------------------------------------------
# Single-phase runner
# ---------------------------------------------------------------------------

def _run_phase(
    dataset_path: str,
    search_space_path: str,
    vary_step: str,
    phase_dir: Path,
    *,
    n_configs: int,
    seed: int,
    think_time_mean: float,
    base_config: ArcoConfig,
    resume: bool,
    enable_codecarbon: bool = False,
    max_prompts: Optional[int] = None,
) -> pd.DataFrame:
    """Run one ablation phase: vary only *vary_step*, others fixed at default.

    Saves per-config artifacts under *phase_dir* and writes detail.csv /
    summary.csv via aggregate_bulk_results.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame for this phase (one row per config).
    """
    phase_dir.mkdir(parents=True, exist_ok=True)

    space = SearchSpace(search_space_path)
    configs: List[ArcoConfig] = space.sample(
        n_configs=n_configs, seed=seed, base_config=base_config, vary_step=vary_step
    )

    # Persist manifest before any run starts
    records = space.configs_to_records(configs)
    with open(phase_dir / "configs_sampled.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Sampled {n_configs} configs → {phase_dir / 'configs_sampled.json'}")
    print()

    think_rng = np.random.RandomState(seed + 1)
    all_results: List[pd.DataFrame] = []

    for config_idx, agent_config in enumerate(configs):
        config_dir = phase_dir / f"config_{config_idx:04d}"

        # Resume: skip if already done
        done_marker = config_dir / "benchmark_results.csv"
        if resume and done_marker.exists():
            print(f"[config {config_idx:04d}] Skipping (already done).")
            df_existing = pd.read_csv(done_marker)
            df_existing["config_id"] = config_idx
            all_results.append(df_existing)
            continue

        config_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        _print_config_summary(config_idx, agent_config, n_configs)
        print(f"{'='*70}")

        with open(config_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(records[config_idx], f, indent=2)

        t_start = time.perf_counter()
        df_config = run_benchmark(
            dataset_path,
            agent_config=agent_config,
            save_dir=str(config_dir),
            save_execution_artifacts=True,
            enable_codecarbon=enable_codecarbon,
            max_prompts=max_prompts,
            config_label=f"config {config_idx + 1}/{n_configs}",
        )
        elapsed = time.perf_counter() - t_start

        df_config["config_id"] = config_idx
        df_config.to_csv(config_dir / "benchmark_results.csv", index=False)
        all_results.append(df_config)

        print(f"\n[config {config_idx:04d}] Done in {elapsed:.1f}s")

        # Think time (skip after last config)
        if config_idx < n_configs - 1 and think_time_mean > 0:
            t_sleep = _exponential_think_time(think_time_mean, think_rng)
            print(f"Think time: sleeping {t_sleep:.1f}s before next config…")
            time.sleep(t_sleep)

    if not all_results:
        print("No results to aggregate.")
        return pd.DataFrame()

    _, summary = aggregate_bulk_results(str(phase_dir), save=True)
    return summary


# ---------------------------------------------------------------------------
# Multi-phase coordinator
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
    vary_step: Optional[str] = None,
    enable_codecarbon: bool = True,
    max_prompts: Optional[int] = None,
) -> pd.DataFrame:
    """Run the 3-phase ablation benchmark.

    By default, runs all three phases sequentially:
      1. vary lookup_sales_data  (other steps fixed at default)
      2. vary analyzing_data     (other steps fixed at default)
      3. vary create_visualization (other steps fixed at default)

    Each phase samples *n_configs* random configs and saves results under
    ``<save_dir>/vary_<step>/``.  At the end a combined ``detail_combined.csv``
    is written to *save_dir*.

    Parameters
    ----------
    dataset_path : str
        Path to the benchmark JSON (e.g. evaluation/benchmark_dataset.json).
    search_space_path : str
        Path to the search space YAML (e.g. evaluation/search_space.yaml).
    n_configs : int
        Number of random configs **per phase** (total runs = 3 × n_configs).
    seed : int
        Master seed for config sampling and think-time draws.
    think_time_mean : float
        Mean of Exponential think-time between runs (seconds; 0 to disable).
    model : str
        LLM model name for all sampled configs.
    provider : str
        LLM provider ("openai" or "ollama").
    openai_api_key : str, optional
        OpenAI API key; defaults to OPENAI_API_KEY env var.
    save_dir : str
        Root directory for all output.
    resume : bool
        Skip configs whose output directory already exists.
    vary_step : str, optional
        Run only this phase (for resuming or debugging a single phase).
        Choices: "lookup_sales_data", "analyzing_data", "create_visualization".
        When None (default), all three phases run in sequence.

    Returns
    -------
    pd.DataFrame
        Combined summary across all phases (one row per config per phase).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    base_config = ArcoConfig(
        model=model,
        provider=provider,
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
    )

    steps_to_run = [vary_step] if vary_step else _VARY_STEPS
    # When running all phases each gets its own subdir; single-phase runs
    # save directly into save_dir to maintain backward compatibility.
    use_subdirs = len(steps_to_run) > 1

    # Separate RNG for inter-phase think times (seed+99 to stay independent
    # from the per-config think RNG used inside each phase, which uses seed+1)
    inter_phase_rng = np.random.RandomState(seed + 99)

    phase_details: List[pd.DataFrame] = []
    phase_summaries: List[pd.DataFrame] = []

    for phase_idx, step in enumerate(steps_to_run):
        phase_dir = save_path / f"vary_{step}" if use_subdirs else save_path

        print(f"\n{'#'*70}")
        print(f"PHASE {phase_idx + 1}/{len(steps_to_run)}: varying '{step}'")
        print(f"Output → {phase_dir}")
        print(f"{'#'*70}")

        summary = _run_phase(
            dataset_path,
            search_space_path,
            step,
            phase_dir,
            n_configs=n_configs,
            seed=seed,
            think_time_mean=think_time_mean,
            base_config=base_config,
            resume=resume,
            enable_codecarbon=enable_codecarbon,
            max_prompts=max_prompts,
        )

        if not summary.empty:
            summary["vary_step"] = step
            phase_summaries.append(summary)

        detail_path = phase_dir / "detail.csv"
        if detail_path.exists():
            df_detail = pd.read_csv(detail_path)
            if "gen_sql" in df_detail.columns:
                df_detail["gen_sql"] = df_detail["gen_sql"].apply(
                    lambda v: " ".join(str(v).split()) if pd.notna(v) else v
                )
            df_detail["vary_step"] = step
            phase_details.append(df_detail)

        _print_phase_summary(summary, step)

        # Think time between phases (not after the last one)
        if use_subdirs and phase_idx < len(steps_to_run) - 1 and think_time_mean > 0:
            inter_phase_sleep = _exponential_think_time(think_time_mean, inter_phase_rng)
            print(f"\nInter-phase pause: sleeping {inter_phase_sleep:.1f}s…")
            time.sleep(inter_phase_sleep)

    # --- Combined output (always generated when there are results) ---
    if phase_details:
        combined_detail = pd.concat(phase_details, ignore_index=True)
        # Reorder: config_id, vary_step, test_case_id, prompt first
        _leading = [c for c in ("config_id", "vary_step", "test_case_id", "prompt", "difficulty") if c in combined_detail.columns]
        _rest = [c for c in combined_detail.columns if c not in set(_leading)]
        combined_detail = combined_detail[_leading + _rest]
        combined_path = save_path / "detail_combined.csv"
        combined_detail.to_csv(combined_path, index=False)
        print(f"\nCombined detail ({len(combined_detail)} rows) → {combined_path}")
        xlsx_path = save_path / "detail_combined.xlsx"
        combined_detail.to_excel(xlsx_path, index=False)
        print(f"Excel version → {xlsx_path}")

    print(f"\n{'#'*70}")
    print("ALL PHASES COMPLETE")
    print(f"Results → {save_dir}")
    print(f"{'#'*70}")

    if not phase_summaries:
        return pd.DataFrame()
    return pd.concat(phase_summaries, ignore_index=True)


def _print_phase_summary(summary: pd.DataFrame, vary_step: str) -> None:
    """Print a short score table for one completed phase."""
    abbr = {"lookup_sales_data": "lsd", "analyzing_data": "ana", "create_visualization": "cvi"}
    print(f"\n--- Phase summary (vary={abbr.get(vary_step, vary_step)}) ---")
    _print_summary_table(summary)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "3-phase ablation bulk runner. "
            "Runs n-configs random configs per phase (total = 3 × n-configs). "
            "Each phase varies one step while the other two stay at default (n=1)."
        )
    )
    parser.add_argument("dataset", help="Path to benchmark dataset JSON")
    parser.add_argument("search_space", help="Path to search_space.yaml")
    parser.add_argument(
        "--n-configs",
        type=int,
        default=3,
        help="Random configs per phase (default: 3; use 50 for full run)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument(
        "--think-time",
        type=float,
        default=5.0,
        help="Mean Exponential think time between runs in seconds (default: 5.0; 0 to disable)",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--provider", default="openai", help="LLM provider (default: openai)")
    parser.add_argument("--save-dir", default="./evaluation/bulk_results", help="Output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs whose output directory already exists",
    )
    parser.add_argument(
        "--vary-step",
        default=None,
        choices=_VARY_STEPS,
        help=(
            "Run only this phase (for resuming or debugging). "
            "Omit to run all three phases in sequence (default behaviour)."
        ),
    )
    parser.add_argument(
        "--no-codecarbon",
        action="store_true",
        help="Disable CodeCarbon energy/emissions tracking (enabled by default)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit to the first N prompts of the benchmark dataset (default: all)",
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
        vary_step=args.vary_step,
        enable_codecarbon=not args.no_codecarbon,
        max_prompts=args.max_prompts,
    )


if __name__ == "__main__":
    main()
