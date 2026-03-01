"""Schema definitions for multi-table database support.

This module provides type-safe dataclass models for describing a database's
tables and columns so the LLM can generate accurate multi-table SQL queries.

Usage example:
    from Agent.schema import ColumnSchema, TableSchema, DatabaseSchema

    schema = DatabaseSchema(tables=[
        TableSchema(
            name="sales",
            description="Daily store-level sales transactions",
            file_path="/data/sales.parquet",
            columns=[
                ColumnSchema(name="Sold_Date", description="Transaction date", data_type="DATE"),
                ColumnSchema(name="Total_Sale_Value", description="Total revenue in USD", data_type="FLOAT"),
            ]
        )
    ])
    print(schema.get_full_schema_str())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ColumnSchema:
    """Description of a single column within a table.

    Attributes:
        name: Column name as it appears in the data file.
        description: Human-readable description for the LLM.
        data_type: SQL data type string (e.g., "VARCHAR", "FLOAT", "DATE", "INTEGER").
        example_values: Optional list of sample values to help the LLM understand the data.
        nullable: Whether the column can contain NULL values.
    """
    name: str
    description: str
    data_type: str = "VARCHAR"
    example_values: Optional[List[str]] = None
    nullable: bool = True


@dataclass
class TableSchema:
    """Description of a single table backed by a parquet file.

    Attributes:
        name: Table name to use in DuckDB (must be a valid SQL identifier).
        description: Human-readable description of what this table contains.
        file_path: Absolute or relative path to the parquet file.
        columns: Ordered list of column definitions.
    """
    name: str
    description: str
    file_path: str
    columns: List[ColumnSchema] = field(default_factory=list)


class DatabaseSchema:
    """Container for all tables in the database, with LLM context helpers.

    When the number of tables exceeds ``compact_threshold``, the agent uses a
    two-step approach: first ask the LLM which tables are relevant (using the
    compact summary), then pass full column details only for the selected tables
    to the SQL generation step. This keeps context windows manageable for 10+
    table schemas.

    Attributes:
        tables: All registered TableSchema objects.
        compact_threshold: Maximum number of tables before switching to the
            two-step table-selection approach. Default is 5.
    """

    def __init__(
        self,
        tables: List[TableSchema],
        compact_threshold: int = 5,
    ) -> None:
        self.tables = tables
        self.compact_threshold = compact_threshold
        self._table_index = {t.name: t for t in tables}

    def get_table(self, name: str) -> Optional[TableSchema]:
        """Return the TableSchema for a given table name, or None if not found."""
        return self._table_index.get(name)

    def should_use_table_selection(self) -> bool:
        """True when the number of tables exceeds compact_threshold."""
        return len(self.tables) > self.compact_threshold

    def get_compact_summary(self) -> str:
        """Return a brief listing of table names and descriptions only.

        Used as first-pass context when there are many tables (> compact_threshold),
        allowing the LLM to select which tables are relevant before being given
        full column details.

        Returns a string like::

            Table: sales
            Description: Daily store-level sales transactions

            Table: products
            Description: Product catalog with pricing information
        """
        lines = []
        for table in self.tables:
            lines.append(f"Table: {table.name}")
            lines.append(f"Description: {table.description}")
            lines.append("")
        return "\n".join(lines).strip()

    def get_full_schema_str(self, table_names: Optional[List[str]] = None) -> str:
        """Return full schema details including column descriptions.

        Args:
            table_names: If provided, only include these tables (preserving the
                order of self.tables). If None, include all tables.

        Returns a string like::

            Table: sales
            Description: Daily store-level sales transactions
            Columns:
              - Sold_Date (DATE): Transaction date
              - Total_Sale_Value (FLOAT): Total revenue in USD [examples: 100.0, 250.5]
              - Store_Number (INTEGER): Unique store identifier [NOT NULL]
        """
        if table_names is not None:
            name_set = set(table_names)
            selected = [t for t in self.tables if t.name in name_set]
        else:
            selected = self.tables

        lines = []
        for table in selected:
            lines.append(f"Table: {table.name}")
            lines.append(f"Description: {table.description}")
            if table.columns:
                lines.append("Columns:")
                for col in table.columns:
                    nullable_str = "" if col.nullable else " [NOT NULL]"
                    col_line = f"  - {col.name} ({col.data_type}): {col.description}{nullable_str}"
                    if col.example_values:
                        examples = ", ".join(str(v) for v in col.example_values[:3])
                        col_line += f" [examples: {examples}]"
                    lines.append(col_line)
            lines.append("")
        return "\n".join(lines).strip()
