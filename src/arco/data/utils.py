import csv
import os
from typing import List, Optional

import pandas as pd
from pandas import DataFrame
from pandas import Series


# -----------------------------
# Utils for DataFrame Management
# -----------------------------

def text_to_csv(text: str) -> List[List[str]]:
    """Convert text table to CSV rows.

    Handles both space-separated and pipe-separated formats.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return []

    rows = []
    for line in lines:
        # Try splitting by multiple spaces first
        if '  ' in line:
            parts = [p.strip() for p in line.split() if p.strip()]
        # Try pipe separator
        elif '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        # Fallback to comma
        else:
            parts = [p.strip() for p in line.split(',') if p.strip()]

        if parts:
            rows.append(parts)

    return rows


def text_to_dataframe(text: str) -> Optional[DataFrame]:
    """Convert text table (from DataFrame.to_string()) back to a pandas DataFrame.

    This function handles the output format from DuckDB query results that have been
    converted to string using df.to_string(). It parses the column-aligned text format.

    Args:
        text: Text table string (space-separated columns with headers).

    Returns:
        pandas DataFrame or None if parsing fails.

    Example input format:
            date  sales  region
        0  2021-11-01    100  North
        1  2021-11-02    150  South
    """
    if not text or not text.strip():
        return None

    try:
        rows = text_to_csv(text)
        if not rows:
            return None

        # Detect index by comparing row lengths
        # If data rows have one more column than header row, it's likely the index
        has_index = False
        if len(rows) > 1:
            # Check if data rows have more columns than header
            if len(rows[1]) > len(rows[0]):
                has_index = True

        if has_index and len(rows) > 0:
            # Header row doesn't have index, use all columns
            # Data rows have index as first element, skip it
            headers = rows[0]
            data_rows = [row[1:] for row in rows[1:] if len(row) > 1]
        else:
            # No index column, first row is headers
            headers = rows[0]
            data_rows = rows[1:]

        if not headers or not data_rows:
            return None

        # Create DataFrame and infer types
        df = pd.DataFrame(data_rows, columns=headers)

        # Try to convert columns to appropriate types
        for col in df.columns:
            try:
                # Try numeric conversion
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Try datetime conversion
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    # Keep as string
                    pass

        return df

    except Exception as e:
        print(f"Error converting text to DataFrame: {e}")
        return None


def save_csv(rows: List[List[str]], filepath: str):
    """Save rows to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def normalize_dataframe_values(df: DataFrame) -> DataFrame:
    """Normalize cell values in a DataFrame for consistent comparison.

    Per-column transformations based on auto-detected type:
    - Numeric integers: cast float-encoded ints (e.g., 33653.0 → 33653)
    - Numeric floats: round to 2 decimal places
    - Dates: format as YYYY-MM-DD string
    - Strings: strip leading/trailing whitespace
    """

    df_copy: DataFrame = df.copy()
    for col in df_copy.columns:
        df_column: pd.Series = df_copy[col]

        numeric: Series = pd.to_numeric(df_column, errors="coerce")
        # Handle datetime dtype first (before numeric, since datetime64 is numeric-castable)
        if pd.api.types.is_datetime64_any_dtype(df_column):
            df_copy[col] = df_column.dt.strftime("%Y-%m-%d")
        elif numeric.notna().all() and not df_column.isna().all(): # Try numeric
            # Check if all values are whole numbers → cast to int
            if (numeric == numeric.round(0)).all():
                df_copy[col] = numeric.astype(int)
            else:
                df_copy[col] = numeric.round(2)
        else:
            # Try datetime (for string columns like "2023-01-01")
            try:
                datetime: Series = pd.to_datetime(df_column, errors="coerce", format="mixed")
                if datetime.notna().all() and not df_column.isna().all():
                    df_copy[col] = datetime.map(lambda x: x.strftime("%Y-%m-%d"))
                    continue
            except TypeError, AttributeError, ValueError:
                pass
            
            df_copy[col] = df_column.astype(str).str.strip()

    return df_copy
