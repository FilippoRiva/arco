#!/usr/bin/env python3
"""Comprehensive IoU test suite for exact-column comparison plus value normalization.

Covers compare_dataframes_iou, normalize_dataframe_values, compare_csv, and
make_csv_evaluator_no_gt with edge cases under the current design, where
column-schema harmonization is expected to happen before IoU.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import (
    compute_csv_iou,
    compare_dataframes_iou,
    make_csv_evaluator_no_gt,
    normalize_dataframe_values,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def check(label: str, score: float, min_expected: float, max_expected: float = 1.0):
    global PASS, FAIL
    ok = min_expected <= score <= max_expected
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
    else:
        PASS += 1
    exp = f"[{min_expected:.2f}–{max_expected:.2f}]"
    print(f"  [{tag}] {label}: {score:.4f}  expected {exp}")


def load_benchmark():
    path = os.path.join(os.path.dirname(__file__), "../output/evaluation/benchmark_dataset.json")
    with open(path) as f:
        return json.load(f)


def gt_to_df(text: str) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.StringIO(text))


# ── 1. compare_dataframes_iou ───────────────────────────────────────────────

def test_exact_match(entries):
    """Baseline: identical DataFrames should always score 1.0."""
    print("\n" + "=" * 70)
    print("1. EXACT MATCH BASELINES")
    print("=" * 70)
    for i, e in enumerate(entries):
        if not e.get("gt_data"):
            continue
        df = gt_to_df(e["gt_data"])
        score = compare_dataframes_iou(df, df.copy())
        check(f"Case {i+1} ({e['prompt'][:40]}...)", score, 1.0)


def test_column_name_variations(entries):
    """Column name aliasing should score low without prior schema standardization."""
    print("\n" + "=" * 70)
    print("2. COLUMN NAME VARIATIONS")
    print("=" * 70)

    # Case 1: partial rename
    df = gt_to_df(entries[0]["gt_data"])
    c = df.copy()
    c.columns = ["Sale_Date", "Total_Sales_Value"]
    check("Partial rename (Sold_Date→Sale_Date)", compare_dataframes_iou(df, c), 0.0, 0.1)

    # Case 2: full rename
    c2 = df.copy()
    c2.columns = ["transaction_date", "revenue"]
    check("Full rename to unrelated names", compare_dataframes_iou(df, c2), 0.0, 0.1)

    # Case 3: case-only difference
    c3 = df.copy()
    c3.columns = ["sold_date", "total_sales_value"]
    check("Case difference only", compare_dataframes_iou(df, c3), 0.0, 0.1)

    # Case 4: UPPER CASE
    c4 = df.copy()
    c4.columns = ["SOLD_DATE", "TOTAL_SALES_VALUE"]
    check("ALL CAPS column names", compare_dataframes_iou(df, c4), 0.0, 0.1)

    # Case 5: 3-column dataset
    df3 = gt_to_df(entries[1]["gt_data"])
    c5 = df3.copy()
    c5.columns = ["Month", "Revenue", "Units_Sold"]
    check("3-col rename (month_start→Month, etc.)", compare_dataframes_iou(df3, c5), 0.0, 0.1)


def test_column_order_variations(entries):
    """Column order changes should score low unless standardization fixed the order first."""
    print("\n" + "=" * 70)
    print("3. COLUMN ORDER VARIATIONS")
    print("=" * 70)

    # 2-column reverse
    df = gt_to_df(entries[0]["gt_data"])
    rev = df[df.columns[::-1]]
    check("2-col reversed order, same names", compare_dataframes_iou(df, rev), 0.0, 0.1)

    # 2-col reverse + rename
    rev2 = rev.copy()
    rev2.columns = ["revenue", "date"]
    check("2-col reversed + renamed", compare_dataframes_iou(df, rev2), 0.0, 0.1)

    # 3-column shuffle
    df3 = gt_to_df(entries[1]["gt_data"])
    shuf = df3[["total_revenue", "total_units", "month_start"]]
    check("3-col shuffled order", compare_dataframes_iou(df3, shuf), 0.0, 0.1)

    # 3-col shuffle + rename
    shuf2 = shuf.copy()
    shuf2.columns = ["Revenue", "Units", "Month"]
    check("3-col shuffled + renamed", compare_dataframes_iou(df3, shuf2), 0.0, 0.1)


def test_value_normalization(entries):
    """Value format differences should be absorbed by normalization."""
    print("\n" + "=" * 70)
    print("4. VALUE FORMAT DIFFERENCES")
    print("=" * 70)

    df = gt_to_df(entries[0]["gt_data"])

    # Date with timestamp
    c1 = df.copy()
    c1["Sold_Date"] = c1["Sold_Date"].astype(str) + " 00:00:00"
    check("Date with ' 00:00:00' suffix", compare_dataframes_iou(df, c1), 0.95)

    # Date with full timestamp
    c1b = df.copy()
    c1b["Sold_Date"] = c1b["Sold_Date"].astype(str) + "T00:00:00"
    check("Date with 'T00:00:00' suffix", compare_dataframes_iou(df, c1b), 0.95)

    # Float rounding
    c2 = df.copy()
    c2["Total_Sales_Value"] = c2["Total_Sales_Value"].round(2)
    check("Floats rounded to 2 decimals", compare_dataframes_iou(df, c2), 0.95)

    # Extra precision
    c3 = df.copy()
    c3["Total_Sales_Value"] = c3["Total_Sales_Value"].apply(lambda x: float(f"{x:.15f}"))
    check("Floats with 15 decimal digits", compare_dataframes_iou(df, c3), 0.95)

    # Float-encoded ints
    df3 = gt_to_df(entries[1]["gt_data"])
    c4 = df3.copy()
    c4["total_units"] = c4["total_units"].astype(float)
    check("Float-encoded ints (33653.0 vs 33653)", compare_dataframes_iou(df3, c4), 0.95)

    # String whitespace
    c5 = df.copy()
    c5["Sold_Date"] = c5["Sold_Date"].astype(str).apply(lambda x: f"  {x}  ")
    check("String values with leading/trailing spaces", compare_dataframes_iou(df, c5), 0.95)


def test_row_order(entries):
    """Row reordering should not affect IoU (greedy matching is order-independent)."""
    print("\n" + "=" * 70)
    print("5. ROW ORDER VARIATIONS")
    print("=" * 70)

    df = gt_to_df(entries[2]["gt_data"])

    # Reverse row order
    rev = df.iloc[::-1].reset_index(drop=True)
    check("Reversed row order", compare_dataframes_iou(df, rev), 0.95)

    # Sorted by a different column
    srt = df.sort_values("Store_Number").reset_index(drop=True)
    check("Sorted by Store_Number (asc)", compare_dataframes_iou(df, srt), 0.95)

    # Random shuffle
    np.random.seed(42)
    shuf = df.sample(frac=1).reset_index(drop=True)
    check("Random shuffle", compare_dataframes_iou(df, shuf), 0.95)


def test_subset_and_superset(entries):
    """Subset/superset columns and rows."""
    print("\n" + "=" * 70)
    print("6. SUBSET / SUPERSET SCENARIOS")
    print("=" * 70)

    df3 = gt_to_df(entries[1]["gt_data"])

    # Missing column (2 vs 3) — compare on exact shared columns only
    sub = df3[["month_start", "total_revenue"]].copy()
    score = compare_dataframes_iou(df3, sub)
    check("Missing 1 column (2 vs 3)", score, 0.5, 1.0)

    # Extra column (4 vs 3)
    sup = df3.copy()
    sup["extra_col"] = range(len(sup))
    score = compare_dataframes_iou(df3, sup)
    check("Extra column (4 vs 3)", score, 0.5, 1.0)

    # Subset of rows (top 6 of 12)
    sub_rows = df3.head(6)
    score = compare_dataframes_iou(df3, sub_rows)
    check("Subset rows (6 of 12)", score, 0.3, 0.6)

    # Extra rows (duplicated)
    dup = pd.concat([df3, df3.head(3)], ignore_index=True)
    score = compare_dataframes_iou(df3, dup)
    check("Extra duplicated rows (15 vs 12)", score, 0.7, 1.0)


def test_completely_different():
    """Completely unrelated data should score near 0."""
    print("\n" + "=" * 70)
    print("7. NEGATIVE CASES")
    print("=" * 70)

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})
    check("Completely different data, same shape", compare_dataframes_iou(df1, df2), 0.0, 0.1)

    df3 = pd.DataFrame({"a": [1]})
    df4 = pd.DataFrame({"x": [10], "y": [20], "z": [30]})
    check("Different col count, no shared names", compare_dataframes_iou(df3, df4), 0.0, 0.1)

    check("None vs DataFrame", compare_dataframes_iou(None, df1), 0.0, 0.0)
    check("Empty vs DataFrame", compare_dataframes_iou(pd.DataFrame(), df1), 0.0, 0.0)


# ── 2. normalize_dataframe_values ───────────────────────────────────────────

def test_normalize():
    """Test normalize_dataframe_values directly."""
    print("\n" + "=" * 70)
    print("8. NORMALIZE_DATAFRAME_VALUES UNIT TESTS")
    print("=" * 70)

    # Mixed types
    df = pd.DataFrame({
        "date_col": ["2021-01-01 00:00:00", "2021-02-15 00:00:00"],
        "float_col": [123.456789, 987.654321],
        "int_col": [100.0, 200.0],
        "str_col": ["  hello  ", "  world  "],
    })
    norm = normalize_dataframe_values(df)

    ok_date = norm["date_col"].iloc[0] == "2021-01-01"
    ok_float = norm["float_col"].iloc[0] == 123.46
    ok_int = norm["int_col"].iloc[0] == 100 and isinstance(norm["int_col"].iloc[0], (int, np.integer))
    ok_str = norm["str_col"].iloc[0] == "hello"

    check("Date normalization (→ YYYY-MM-DD)", 1.0 if ok_date else 0.0, 1.0)
    check("Float rounding (→ 2dp)", 1.0 if ok_float else 0.0, 1.0)
    check("Int casting (100.0 → 100)", 1.0 if ok_int else 0.0, 1.0)
    check("String stripping", 1.0 if ok_str else 0.0, 1.0)

    # Edge: None/empty
    check("None input returns None", 1.0 if normalize_dataframe_values(None) is None else 0.0, 1.0)
    empty = normalize_dataframe_values(pd.DataFrame())
    check("Empty input returns empty", 1.0 if empty.empty else 0.0, 1.0)


# ── 3. compare_csv (legacy) ────────────────────────────────────────────────

def test_compare_csv(entries):
    """Test the legacy compare_csv function with file-based inputs."""
    print("\n" + "=" * 70)
    print("9. COMPARE_CSV (LEGACY FILE-BASED)")
    print("=" * 70)

    df = gt_to_df(entries[2]["gt_data"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
        df.to_csv(f1, index=False)
        path1 = f1.name

    # Exact match
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
        df.to_csv(f2, index=False)
        path2 = f2.name

    col_iou, row_iou, data_iou = compute_csv_iou(path1, path2)
    check("Exact CSV: col_names_iou", col_iou, 1.0)
    check("Exact CSV: rows_iou", row_iou, 1.0)
    check("Exact CSV: data_iou", data_iou, 1.0)

    # Case difference
    df_case = df.copy()
    df_case.columns = ["store_number", "Total_Revenue"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f3:
        df_case.to_csv(f3, index=False)
        path3 = f3.name

    col_iou2, row_iou2, data_iou2 = compute_csv_iou(path1, path3)
    check("Case-diff CSV: col_names_iou", col_iou2, 0.4, 1.0)
    check("Case-diff CSV: data_iou", data_iou2, 0.5, 1.0)

    # Clean up
    for p in [path1, path2, path3]:
        os.unlink(p)


# ── 4. make_csv_evaluator_no_gt (consensus) ────────────────────────────────

def test_consensus_evaluator(entries):
    """Test best-of-n consensus scoring with varied candidates."""
    print("\n" + "=" * 70)
    print("10. CONSENSUS EVALUATOR (NO-GT BEST-OF-N)")
    print("=" * 70)

    batch_eval = make_csv_evaluator_no_gt()
    df = gt_to_df(entries[2]["gt_data"])

    # 3 candidates: same data, different column names/order
    c1 = df.copy()
    c2 = df.copy()
    c2.columns = ["store_number", "Total_Revenue"]
    c3 = df[df.columns[::-1]].copy()
    c3.columns = ["revenue", "store_id"]

    results = [
        {"data_df": c1, "sql_query": "SELECT Store_Number, total_revenue ..."},
        {"data_df": c2, "sql_query": "SELECT store_number, Total_Revenue ..."},
        {"data_df": c3, "sql_query": "SELECT total_revenue AS revenue, Store_Number AS store_id ..."},
    ]
    scores = batch_eval(results, {})
    print(f"  Consensus scores: {[f'{s:.4f}' for s in scores]}")
    check("All candidates should score high (mean)", np.mean(scores), 0.8)
    check("Score variance should be low", 1.0 - np.std(scores), 0.8)

    # 3 candidates: 2 agree, 1 outlier
    outlier = pd.DataFrame({"Store_Number": [9999], "total_revenue": [1.0]})
    results2 = [
        {"data_df": c1},
        {"data_df": c2},
        {"data_df": outlier},
    ]
    scores2 = batch_eval(results2, {})
    print(f"  Scores with outlier: {[f'{s:.4f}' for s in scores2]}")
    check("Outlier should score lowest", 1.0 if scores2[2] < scores2[0] else 0.0, 1.0)

    # Single candidate
    scores3 = batch_eval([{"data_df": c1}], {})
    check("Single candidate → score 1.0", scores3[0], 1.0)


# ── 5. Combined stress test ─────────────────────────────────────────────────

def test_combined_stress(entries):
    """Everything at once: different names, order, case, value formats."""
    print("\n" + "=" * 70)
    print("11. COMBINED STRESS TEST (ALL VARIATIONS AT ONCE)")
    print("=" * 70)

    df = gt_to_df(entries[0]["gt_data"])

    # Build a maximally-different-looking but semantically identical candidate
    # (using round(2) — realistic SQL precision, not round(1) which creates >atol gaps)
    stress = df[df.columns[::-1]].copy()  # reverse order
    stress.columns = ["REVENUE", "DATE"]  # different names + case
    stress["DATE"] = stress["DATE"].astype(str) + " 00:00:00"  # timestamp suffix
    stress["REVENUE"] = stress["REVENUE"].round(2)  # realistic SQL precision
    stress = stress.iloc[::-1].reset_index(drop=True)  # reverse rows

    score = compare_dataframes_iou(df, stress)
    check("Max stress: reversed cols + renamed + timestamps + round(2) + reversed rows", score, 0.90)

    # Extreme: round(1) creates >0.01 gaps — should still get partial credit
    stress_extreme = df[df.columns[::-1]].copy()
    stress_extreme.columns = ["REVENUE", "DATE"]
    stress_extreme["DATE"] = stress_extreme["DATE"].astype(str) + " 00:00:00"
    stress_extreme["REVENUE"] = stress_extreme["REVENUE"].round(1)
    stress_extreme = stress_extreme.iloc[::-1].reset_index(drop=True)

    score_ext = compare_dataframes_iou(df, stress_extreme)
    check("Extreme stress with round(1) — partial match expected", score_ext, 0.0, 0.5)

    # 3-column version (realistic precision)
    df3 = gt_to_df(entries[1]["gt_data"])
    stress3 = df3[["total_units", "month_start", "total_revenue"]].copy()
    stress3.columns = ["Units", "Month", "Rev"]
    stress3["Month"] = stress3["Month"].astype(str) + "T00:00:00"
    stress3["Rev"] = stress3["Rev"].round(2)  # realistic
    stress3["Units"] = stress3["Units"].astype(float)
    stress3 = stress3.sample(frac=1, random_state=99).reset_index(drop=True)

    score3 = compare_dataframes_iou(df3, stress3)
    check("3-col stress: shuffled + renamed + timestamps + round(2) + shuffled rows", score3, 0.90)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    entries = load_benchmark()

    test_exact_match(entries)
    test_column_name_variations(entries)
    test_column_order_variations(entries)
    test_value_normalization(entries)
    test_row_order(entries)
    test_subset_and_superset(entries)
    test_completely_different()
    test_normalize()
    test_compare_csv(entries)
    test_consensus_evaluator(entries)
    test_combined_stress(entries)

    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {FAIL} tests need attention")
    print("=" * 70)
