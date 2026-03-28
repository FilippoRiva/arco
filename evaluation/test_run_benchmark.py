#!/usr/bin/env python3
"""Test suite for evaluation/run_benchmark.py and the refactored evaluator wiring.

Covers:
  1. load_benchmark_dataset validation
  2. make_csv_evaluator_gt with ground_truth_csv_text
  3. run_benchmark eval function wiring (mocked agent)
  4. CSV GT evaluator integration
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agent.utils import make_csv_evaluator_gt
from evaluation.run_benchmark import load_benchmark_dataset, run_benchmark

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def check(label: str, score, min_expected: float, max_expected: float = None):
    """Check that score is within [min_expected, max_expected]."""
    global PASS, FAIL
    if max_expected is None:
        max_expected = min_expected
    ok = min_expected <= score <= max_expected
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    exp = f"[{min_expected:.2f}–{max_expected:.2f}]"
    print(f"  [{tag}] {label}: {score}  expected {exp}")


def check_raises(label: str, exc_type, fn, *args, substr: str = None, **kwargs):
    """Check that fn(*args) raises exc_type, optionally with substr in message."""
    global PASS, FAIL
    try:
        fn(*args, **kwargs)
        FAIL += 1
        print(f"  [FAIL] {label}: no exception raised")
    except exc_type as e:
        if substr and substr not in str(e):
            FAIL += 1
            print(f"  [FAIL] {label}: raised {exc_type.__name__} but missing '{substr}' in: {e}")
        else:
            PASS += 1
            print(f"  [PASS] {label}: correctly raised {exc_type.__name__}")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {label}: wrong exception {type(e).__name__}: {e}")


def load_benchmark():
    path = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    with open(path) as f:
        return json.load(f)


def gt_to_df(text: str) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.StringIO(text))


# ── 1. load_benchmark_dataset ───────────────────────────────────────────────

def test_load_benchmark_dataset():
    print("\n" + "=" * 70)
    print("1. load_benchmark_dataset VALIDATION")
    print("=" * 70)

    # 1a. Valid dataset
    path = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    entries = load_benchmark_dataset(path)
    check("Valid dataset returns entries", len(entries), 3, 3)
    check("Each entry has prompt", all("prompt" in e for e in entries), 1, 1)

    # 1b. Empty dataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([], f)
        f.flush()
        check_raises("Empty dataset", ValueError, load_benchmark_dataset, f.name, substr="Empty dataset")
    os.unlink(f.name)

    # 1c. Missing prompt field
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"gt_data": "a,b\n1,2"}], f)
        f.flush()
        check_raises("Missing prompt", ValueError, load_benchmark_dataset, f.name, substr="prompt")
    os.unlink(f.name)

    # 1d. Missing gt fields
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"prompt": "x"}], f)
        f.flush()
        check_raises("Missing gt fields", ValueError, load_benchmark_dataset, f.name, substr="gt_data")
    os.unlink(f.name)


# ── 2. make_csv_evaluator_gt with ground_truth_csv_text ─────────────────────

def test_csv_evaluator_gt_text():
    print("\n" + "=" * 70)
    print("2. make_csv_evaluator_gt WITH ground_truth_csv_text")
    print("=" * 70)

    entries = load_benchmark()
    csv_text = entries[0]["gt_data"]
    gt_df = gt_to_df(csv_text)

    # 2a. Exact match via text input
    eval_fn = make_csv_evaluator_gt(ground_truth_csv_text=csv_text)
    score = eval_fn({"data_df": gt_df.copy()}, {})
    check("Text input exact match", score, 1.0, 1.0)

    # 2b. Different data scores lower
    different_df = gt_df.copy()
    different_df.iloc[0, -1] = 999999.99
    score = eval_fn({"data_df": different_df}, {})
    check("Text input different data", score, 0.0, 0.99)

    # 2c. None data_df returns 0
    score = eval_fn({"data_df": None}, {})
    check("None data_df returns 0", score, 0.0, 0.0)

    # 2d. Both args raises ValueError
    check_raises(
        "Both args raises ValueError",
        ValueError,
        make_csv_evaluator_gt,
        ground_truth_csv_path="/tmp/fake.csv",
        ground_truth_csv_text=csv_text,
        substr="not both",
    )

    # 2e. Neither arg raises ValueError
    check_raises(
        "Neither arg raises ValueError",
        ValueError,
        make_csv_evaluator_gt,
        substr="Provide either",
    )


# ── 3. run_benchmark eval wiring (mocked agent) ────────────────────────────

def _make_fake_result(entry):
    """Build a fake agent result dict from a benchmark entry."""
    gt_df = gt_to_df(entry["gt_data"]) if entry.get("gt_data") else None
    result = {
        "sql_query": entry.get("gt_sql", ""),
        "data_df": gt_df,
        "data": entry.get("gt_data", ""),
        "answer": [entry.get("gt_analysis", "")],
    }
    if entry.get("gt_chart_config"):
        result["chart_config"] = entry["gt_chart_config"]
        if entry.get("gt_chart_code"):
            result["answer"].append(entry["gt_chart_code"])
    return result


def test_run_benchmark_wiring():
    print("\n" + "=" * 70)
    print("3. run_benchmark EVAL FUNCTION WIRING")
    print("=" * 70)

    entries = load_benchmark()

    # 3a. Data-only entry (case 3: top 5 stores, no vis)
    data_only = [entries[2]]  # gt_chart_config is null
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data_only, f)
        dataset_path = f.name

    with tempfile.TemporaryDirectory() as save_dir:
        fake_result = _make_fake_result(data_only[0])

        with patch("evaluation.run_benchmark.SalesDataAgent") as MockAgent, \
             patch("evaluation.run_benchmark.judge_analysis", return_value=(0.85, {"reasoning": "mocked"})):
            instance = MockAgent.return_value
            instance.run.return_value = fake_result

            df = run_benchmark(dataset_path, save_dir=save_dir)

            check("Data-only: csv_iou present", df["csv_iou"].notna().all(), 1, 1)
            check("Data-only: text_score present", df["text_score"].notna().all(), 1, 1)
            check("Data-only: vis_score is None", df["vis_score"].isna().all(), 1, 1)

            # Verify CSV was saved
            csv_path = os.path.join(save_dir, "benchmark_results.csv")
            check("Results CSV exists", os.path.exists(csv_path), 1, 1)

    os.unlink(dataset_path)

    # 3b. Data+vis entry (case 1: Nov 2021 sales)
    data_vis = [entries[0]]  # has gt_chart_config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data_vis, f)
        dataset_path = f.name

    with tempfile.TemporaryDirectory() as save_dir:
        fake_result = _make_fake_result(data_vis[0])

        with patch("evaluation.run_benchmark.SalesDataAgent") as MockAgent, \
             patch("evaluation.run_benchmark.judge_analysis", return_value=(0.85, {"reasoning": "mocked"})), \
             patch("evaluation.run_benchmark.judge_visualization", return_value=(0.9, {"reasoning": "mocked"})):
            instance = MockAgent.return_value
            instance.run.return_value = fake_result

            df = run_benchmark(dataset_path, save_dir=save_dir)

            check("Data+vis: csv_iou present", df["csv_iou"].notna().all(), 1, 1)
            check("Data+vis: text_score present", df["text_score"].notna().all(), 1, 1)
            check("Data+vis: vis_score present", df["vis_score"].notna().all(), 1, 1)
            check("Data+vis: vis_score is 0.9 (mocked)", float(df["vis_score"].iloc[0]), 0.9, 0.9)

    os.unlink(dataset_path)


# ── 4. CSV GT evaluator integration ─────────────────────────────────────────

def test_csv_gt_evaluator_integration():
    print("\n" + "=" * 70)
    print("4. CSV GT EVALUATOR INTEGRATION")
    print("=" * 70)

    entries = load_benchmark()

    # 4a. Exact match for each entry with gt_data
    for i, entry in enumerate(entries):
        if not entry.get("gt_data"):
            continue
        eval_fn = make_csv_evaluator_gt(ground_truth_csv_text=entry["gt_data"])
        gt_df = gt_to_df(entry["gt_data"])
        score = eval_fn({"data_df": gt_df.copy()}, {})
        check(f"Case {i+1} exact match via text", score, 1.0, 1.0)

    # 4b. Text fallback (pass raw CSV text as "data" instead of data_df)
    entry = entries[2]  # top 5 stores — simpler data
    eval_fn = make_csv_evaluator_gt(ground_truth_csv_text=entry["gt_data"])
    score = eval_fn({"data_df": None, "data": entry["gt_data"]}, {})
    check("Text fallback parses and scores", score, 0.5, 1.0)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("TEST SUITE: evaluation/run_benchmark.py")
    print("=" * 70)

    test_load_benchmark_dataset()
    test_csv_evaluator_gt_text()
    test_run_benchmark_wiring()
    test_csv_gt_evaluator_integration()

    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 70)

    sys.exit(1 if FAIL > 0 else 0)
