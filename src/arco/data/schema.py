"""Schema definitions for multi-table database support.

This module provides type-safe dataclass models for describing a database's
tables and columns so the LLM can generate accurate multi-table SQL queries.

Usage example:
    from arco.schema import ColumnSchema, TableSchema, DatabaseSchema

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
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from arco.data.exceptions import SchemaParsingException


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
    example_values: list[str] | None = None
    nullable: bool = True

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "example_values": self.example_values,
            "nullable": self.nullable,
        }


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
    columns: list[ColumnSchema] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "columns": {column.name: column.to_dict() for column in self.columns},
        }


@dataclass
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

    tables: list[TableSchema] = field(default_factory=list)
    compact_threshold: int = 5
    _table_index: dict[str, TableSchema] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Initialize the index once tables are provided
        if self.tables:
            self._table_index = {table.name: table for table in self.tables}
        else:
            self.tables = []

    def get_table(self, name: str) -> TableSchema | None:
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

    def get_full_schema_str(self, table_names: list[str] | None = None) -> str:
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
                        examples = ", ".join(v for v in col.example_values[:3])
                        col_line += f" [examples: {examples}]"
                    lines.append(col_line)
            lines.append("")
        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            table_name: table_schema.to_dict()
            for (table_name, table_schema) in self._table_index.items()
        }

    @classmethod
    def from_data_dir(cls, data_dir_path: str) -> DatabaseSchema:
        from glob import glob

        import yaml

        data_dir = os.path.abspath(data_dir_path)

        schema_files = sorted(glob(os.path.join(data_dir, "*_schema.yaml")))
        tables = []
        for table_path in schema_files:
            with open(table_path, "r") as tf:
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

            # Resolve file_path relative to schema file directory
            schema_dir = os.path.dirname(table_path)
            file_path = t["file_path"]
            if not os.path.isabs(file_path):
                file_path = os.path.join(schema_dir, file_path)

            tables.append(
                TableSchema(
                    name=t["name"],
                    description=t.get("description", t["name"]),
                    file_path=file_path,
                    columns=columns,
                )
            )

        if tables:
            schema = cls(tables=tables, compact_threshold=5)
            return schema
        else:
            raise SchemaParsingException("The schema was not parsable")
