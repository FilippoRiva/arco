#!/usr/bin/env python3
"""Diagnostic script to test current IoU behavior with synthetic column/value variations.

The current design expects column-schema harmonization to happen before IoU,
for example via LLM-based standardization across best-of-N candidates. This
script highlights that compare_dataframes_iou only tolerates value-format
differences on already-aligned columns.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils import compare_dataframes_iou, normalize_dataframe_values


def load_benchmark():
    path = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    with open(path) as f:
        return json.load(f)


def gt_to_df(gt_data_text: str) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.StringIO(gt_data_text))


def print_result(label: str, score: float, expected: str = ""):
    status = "OK" if score > 0.9 else "LOW"
    exp = f"  (expected: {expected})" if expected else ""
    print(f"  [{status}] {label}: {score:.4f}{exp}")


def test_case_1(entries):
    """Sales in Nov 2021 — GT columns: Sold_Date, Total_Sales_Value"""
    print("\n" + "=" * 70)
    print("TEST CASE 1: Sales in Nov 2021")
    print("GT columns: Sold_Date, Total_Sales_Value")
    print("=" * 70)

    gt_df = gt_to_df(entries[0]["gt_data"])
    print(f"GT shape: {gt_df.shape}, columns: {list(gt_df.columns)}")

    # Test A: Exact match (baseline)
    score = compare_dataframes_iou(gt_df, gt_df.copy())
    print_result("Exact match (baseline)", score, "1.0")

    # Test B: Different column names, same data & order
    candidate_b = gt_df.copy()
    candidate_b.columns = ["Sale_Date", "Total_Sales_Value"]
    score = compare_dataframes_iou(gt_df, candidate_b)
    print_result("Different col name (Sold_Date→Sale_Date), same order", score, "LOW without prior LLM standardization")

    # Test C: Completely different column names, same data & order
    candidate_c = gt_df.copy()
    candidate_c.columns = ["transaction_date", "revenue"]
    score = compare_dataframes_iou(gt_df, candidate_c)
    print_result("Completely different col names, same order", score, "LOW without prior LLM standardization")

    # Test D: Same data, reversed column order
    candidate_d = gt_df[gt_df.columns[::-1]].copy()
    score = compare_dataframes_iou(gt_df, candidate_d)
    print_result("Reversed column order, same names", score, "LOW unless order was standardized first")

    # Test E: Different names AND reversed order
    candidate_e = gt_df[gt_df.columns[::-1]].copy()
    candidate_e.columns = ["revenue", "transaction_date"]
    score = compare_dataframes_iou(gt_df, candidate_e)
    print_result("Different names + reversed order", score, "LOW without prior LLM standardization")

    # Test F: Date values with " 00:00:00" suffix
    candidate_f = gt_df.copy()
    candidate_f.iloc[:, 0] = candidate_f.iloc[:, 0].astype(str) + " 00:00:00"
    score = compare_dataframes_iou(gt_df, candidate_f)
    print_result("Dates with ' 00:00:00' suffix", score, "~1.0")

    # Test G: Case difference in column names
    candidate_g = gt_df.copy()
    candidate_g.columns = ["sold_date", "total_sales_value"]
    score = compare_dataframes_iou(gt_df, candidate_g)
    print_result("Lowercase column names", score, "LOW without prior LLM standardization")

    # Test H: Float precision difference
    candidate_h = gt_df.copy()
    candidate_h["Total_Sales_Value"] = candidate_h["Total_Sales_Value"].round(2)
    score = compare_dataframes_iou(gt_df, candidate_h)
    print_result("Floats rounded to 2 decimals", score, "~1.0")


def test_case_2(entries):
    """12 months of 2023 — GT columns: month_start, total_revenue, total_units"""
    print("\n" + "=" * 70)
    print("TEST CASE 2: 12 months of 2023 revenue")
    print("GT columns: month_start, total_revenue, total_units")
    print("=" * 70)

    gt_df = gt_to_df(entries[1]["gt_data"])
    print(f"GT shape: {gt_df.shape}, columns: {list(gt_df.columns)}")

    # Test A: Exact match
    score = compare_dataframes_iou(gt_df, gt_df.copy())
    print_result("Exact match (baseline)", score, "1.0")

    # Test B: Different column names, same order
    candidate_b = gt_df.copy()
    candidate_b.columns = ["Month", "Revenue", "Units_Sold"]
    score = compare_dataframes_iou(gt_df, candidate_b)
    print_result("Different col names (Month/Revenue/Units_Sold)", score, "LOW without prior LLM standardization")

    # Test C: Same names, shuffled column order
    candidate_c = gt_df[["total_revenue", "month_start", "total_units"]].copy()
    score = compare_dataframes_iou(gt_df, candidate_c)
    print_result("Shuffled column order (rev, month, units)", score, "LOW unless order was standardized first")

    # Test D: Different names AND shuffled order
    candidate_d = gt_df[["total_revenue", "month_start", "total_units"]].copy()
    candidate_d.columns = ["Revenue", "Month", "Units"]
    score = compare_dataframes_iou(gt_df, candidate_d)
    print_result("Different names + shuffled order", score, "LOW without prior LLM standardization")

    # Test E: Missing one column (2 cols vs 3 cols)
    candidate_e = gt_df[["month_start", "total_revenue"]].copy()
    score = compare_dataframes_iou(gt_df, candidate_e)
    print_result("Missing one column (2 vs 3)", score, "should be partial")

    # Test F: Float-encoded ints (33653.0 vs 33653)
    candidate_f = gt_df.copy()
    candidate_f["total_units"] = candidate_f["total_units"].astype(float)
    score = compare_dataframes_iou(gt_df, candidate_f)
    print_result("Float-encoded ints (33653.0 vs 33653)", score, "~1.0")


def test_case_3(entries):
    """Top 5 stores — GT columns: Store_Number, total_revenue"""
    print("\n" + "=" * 70)
    print("TEST CASE 3: Top 5 stores by revenue")
    print("GT columns: Store_Number, total_revenue")
    print("=" * 70)

    gt_df = gt_to_df(entries[2]["gt_data"])
    print(f"GT shape: {gt_df.shape}, columns: {list(gt_df.columns)}")

    # Test A: Exact match
    score = compare_dataframes_iou(gt_df, gt_df.copy())
    print_result("Exact match (baseline)", score, "1.0")

    # Test B: Case difference
    candidate_b = gt_df.copy()
    candidate_b.columns = ["store_number", "Total_Revenue"]
    score = compare_dataframes_iou(gt_df, candidate_b)
    print_result("Case difference (store_number/Total_Revenue)", score, "LOW without prior LLM standardization")

    # Test C: Reversed column order
    candidate_c = gt_df[gt_df.columns[::-1]].copy()
    score = compare_dataframes_iou(gt_df, candidate_c)
    print_result("Reversed column order", score, "LOW unless order was standardized first")

    # Test D: Different row order (sorted differently)
    candidate_d = gt_df.sort_values("Store_Number").reset_index(drop=True)
    score = compare_dataframes_iou(gt_df, candidate_d)
    print_result("Different row order (sorted by Store_Number)", score, "~1.0")

    # Test E: Rounded revenue values
    candidate_e = gt_df.copy()
    candidate_e["total_revenue"] = candidate_e["total_revenue"].round(2)
    score = compare_dataframes_iou(gt_df, candidate_e)
    print_result("Revenue rounded to 2 decimals", score, "~1.0")


def test_normalize(entries):
    """Test normalize_dataframe_values on various value formats."""
    print("\n" + "=" * 70)
    print("VALUE NORMALIZATION TESTS")
    print("=" * 70)

    gt_df = gt_to_df(entries[0]["gt_data"])

    # Test: date with timestamp vs clean date
    candidate = gt_df.copy()
    candidate["Sold_Date"] = candidate["Sold_Date"].astype(str) + " 00:00:00"
    candidate["Total_Sales_Value"] = candidate["Total_Sales_Value"].apply(
        lambda x: float(f"{x:.10f}")  # extra decimal precision
    )
    norm = normalize_dataframe_values(candidate)
    print(f"  Before normalize: dates like '{candidate['Sold_Date'].iloc[0]}', "
          f"floats like {candidate['Total_Sales_Value'].iloc[0]}")
    print(f"  After normalize:  dates like '{norm['Sold_Date'].iloc[0]}', "
          f"floats like {norm['Total_Sales_Value'].iloc[0]}")

    score = compare_dataframes_iou(gt_df, candidate)
    print_result("Timestamp dates + extra float precision", score, "~1.0")

    # Test: float-encoded ints
    gt_df2 = gt_to_df(entries[1]["gt_data"])
    candidate2 = gt_df2.copy()
    candidate2["total_units"] = candidate2["total_units"].astype(float)
    norm2 = normalize_dataframe_values(candidate2)
    print(f"  Float int before: {candidate2['total_units'].iloc[0]} (type: {type(candidate2['total_units'].iloc[0]).__name__})")
    print(f"  Float int after:  {norm2['total_units'].iloc[0]} (type: {type(norm2['total_units'].iloc[0]).__name__})")

    score = compare_dataframes_iou(gt_df2, candidate2)
    print_result("Float-encoded ints with normalization", score, "~1.0")


def test_cross_candidate():
    """Simulate no-GT consensus before LLM standardization is applied."""
    print("\n" + "=" * 70)
    print("CROSS-CANDIDATE TEST: Simulated best-of-3 consensus")
    print("=" * 70)

    # Base data (from benchmark case 3)
    data = {
        "Store_Number": [2970, 3300, 1320, 1650, 1210],
        "total_revenue": [836341.33, 619660.17, 592832.07, 580443.01, 508393.77],
    }

    # 3 candidates with different column naming
    c1 = pd.DataFrame(data)
    c1.columns = ["Store_Number", "total_revenue"]

    c2 = pd.DataFrame(data)
    c2.columns = ["store_number", "Total_Revenue"]

    c3_data = {"total_revenue": data["total_revenue"], "Store_Number": data["Store_Number"]}
    c3 = pd.DataFrame(c3_data)  # reversed column order

    pairs = [("C1 vs C2", c1, c2), ("C1 vs C3", c1, c3), ("C2 vs C3", c2, c3)]
    for label, a, b in pairs:
        score = compare_dataframes_iou(a, b)
        print_result(
            f"{label} (cols: {list(a.columns)} vs {list(b.columns)})",
            score,
            "LOW before LLM standardization",
        )


if __name__ == "__main__":
    entries = load_benchmark()
    test_case_1(entries)
    test_case_2(entries)
    test_case_3(entries)
    test_cross_candidate()

    test_normalize(entries)

    print("\n" + "=" * 70)
    print("SUMMARY: [LOW] is expected for schema mismatches unless LLM standardization ran first")
    print("=" * 70)
