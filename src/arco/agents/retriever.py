import json
from copy import deepcopy
from typing import TYPE_CHECKING

import duckdb
import pandas as pd
from langchain_core.language_models import BaseChatModel

from arco.core import Agent, AgentType, Answer, llm_tools
from arco.core.agent import AgentException
from arco.data import DatabaseSchema, normalize_dataframe_values
from arco.evaluators import RetrieverEvaluator

if TYPE_CHECKING:
    from arco.core import AgentConfig, Evaluator, State
    from arco.core.llm_tools import CoTRefiner
    from arco.core.tracking import LLMCallAccumulator


class Retriever(Agent):
    _TABLE_SELECTION_PROMPT = """You are a database architect helping identify which tables are needed to answer a user's question.

## TASK
From the list of available tables, select only the tables needed to answer the user's question.

## AVAILABLE TABLES
{compact_schema}

## USER QUESTION
{prompt}

## CHAIN OF THOUGHT REASONING
Before selecting tables, think step by step:

**Step 1: Understanding the Question**
- What is the user really asking for?
- What entities or concepts are mentioned? (e.g., products, sales, customers, dates)
- What metrics or dimensions does the answer require?

**Step 2: Mapping Concepts to Tables**
- Which table descriptions match the entities mentioned in the question?
- Is the question asking about relationships between multiple entities (implies a JOIN)?
- Are any tables clearly irrelevant (different domain, different subject)?

**Step 3: Identifying Required Joins**
- If multiple entities are needed, which tables contain them?
- Do any tables serve as lookup/dimension tables needed to label results?
- Is there a fact table that connects the needed entities?

**Step 4: Checking Completeness**
- Do the selected tables together contain all the data needed to answer the question?
- Is any additional table needed for filtering or context?
- Are there redundant tables containing the same data?

**Step 5: Final Selection**
- List only the table names that are necessary and sufficient to answer the question
- When in doubt, include a table rather than exclude it (extra context is better than missing data)
- Use only table names exactly as listed in AVAILABLE TABLES

## OUTPUT FORMAT
Return ONLY a comma-separated list of table names. No explanations. No markdown. Just table names.
Example: sales, products
"""

    _SQL_GENERATION_PROMPT = """You are an expert SQL developer specializing in DuckDB queries for data analysis and visualization.

## TASK
Generate a DuckDB SQL query to answer the user's question and provide data optimized for analysis and visualization.

## AVAILABLE DATA
{schema_context}

## USER QUESTION
- prompt : {prompt}

## INSTRUCTIONS
1. Analyze the user's question to identify what data is needed
2. Consider the visualization goal to structure the query output appropriately
3. Select appropriate columns from the schema above
4. Use proper SQL syntax for filtering, aggregation, sorting, and joins across tables
5. Handle NULL values appropriately
6. Use DuckDB-specific functions when beneficial
7. **When using JOINs**: always qualify every column reference with its table alias (e.g. `st.region`, not `region`). In SELECT, GROUP BY, ORDER BY, and WHERE, prefix each column with the correct alias of the table it belongs to. Never reference a column by name alone when multiple tables are in scope.

## QUERY OPTIMIZATION FOR VISUALIZATION
- **For time series plots**: Ensure dates are sorted chronologically, use DATE_TRUNC for proper granularity
- **For bar charts**: Aggregate data by category, order by the metric being compared
- **For scatter plots**: Select two numeric columns that show relationships
- **For trend analysis**: Include time-based grouping (daily, monthly, yearly)
- **General**: Limit result size if needed, ensure clean column names for axis labels

## KEY PITFALLS (MUST AVOID)
- **NEVER use SUBSTR() or SUBSTRING() directly on a DATE column** — DuckDB DATE columns are not strings. Use YEAR() / EXTRACT() instead.
- **NEVER use LIKE directly on a DATE column** — cast to VARCHAR first: `CAST(date_col AS VARCHAR) LIKE '2023%'`
- **NEVER use strptime() on a column that is already DATE type** — it expects a string input.
- **NEVER use strftime(date, format)** — that is SQLite argument order. DuckDB does not support it. Use `YEAR()` or `EXTRACT()` instead.
- **For "average of aggregates" pattern** (e.g., "average monthly revenue"): First aggregate raw rows to the desired period using SUM + GROUP BY, then wrap in a subquery and apply AVG. Do NOT apply AVG directly to individual transaction values.

## EXAMPLES

Example 1 - Simple date filter with aggregation:
    Question: "Show me sales from November 2021"
    Query: SELECT Sold_Date, SUM(Total_Sale_Value) as Total_Revenue FROM sales WHERE CAST(Sold_Date AS VARCHAR) LIKE '%2021-11%' GROUP BY Sold_Date ORDER BY Sold_Date

Example 2 - Multi-table JOIN:
    Question: "Show total revenue by product category for 2023"
    Schema: Table: sales (Sold_Date, SKU_Coded, Total_Sale_Value); Table: products (SKU_Coded, Category, Product_Name)
    Query: SELECT p.Category, SUM(s.Total_Sale_Value) as Total_Revenue FROM sales s JOIN products p ON s.SKU_Coded = p.SKU_Coded WHERE EXTRACT(YEAR FROM s.Sold_Date) = 2023 GROUP BY p.Category ORDER BY Total_Revenue DESC

Example 3 - Two-level aggregation ("average monthly revenue"):
    Question: "Compare average monthly revenue between store regions for 2022 and 2023"
    Schema: Table: sales (Sold_Date, Store_Number, Total_Sale_Value); Table: stores (Store_Number, region)
    Query: SELECT st.region, s.yr AS year, ROUND(AVG(s.monthly_rev), 2) AS avg_monthly_revenue FROM (SELECT Store_Number, YEAR(CAST(Sold_Date AS DATE)) AS yr, DATE_TRUNC('month', CAST(Sold_Date AS DATE)) AS month, SUM(Total_Sale_Value) AS monthly_rev FROM sales WHERE YEAR(CAST(Sold_Date AS DATE)) IN (2022, 2023) GROUP BY Store_Number, yr, month) s JOIN stores st ON s.Store_Number = st.Store_Number GROUP BY st.region, s.yr ORDER BY st.region, s.yr

## OUTPUT FORMAT
Return ONLY the SQL query as plain text. No explanations. No markdown formatting. No code fences. Just the SQL query.
"""

    _COLUMN_STANDARDIZATION_PROMPT = """\
You are a data schema expert. Given N SQL queries against the same database that \
answer the same question, standardize their result column names and order.

## Database Schema
{schema_context}

## Candidates
{candidates_section}

## Rules
- For columns that come directly from schema tables, use the exact schema column name.
- For aggregated/computed columns (SUM, COUNT, AVG, etc.), pick the most descriptive \
name used by any candidate. Prefer lowercase_with_underscores.
- All candidates MUST map to the same canonical columns in the same order.
- Return ONLY valid JSON, no explanation or markdown fences.

## Output format
{{"canonical_columns": ["col1", "col2"], "mappings": [{{"original_col": "canonical_col", ...}}, ...]}}
"""

    def __init__(self, data_dir: str | None = None):
        super().__init__()
        self.schema: DatabaseSchema = DatabaseSchema.from_data_dir(data_dir or "./data")

    @staticmethod
    def _select_relevant_tables(
        state: State, schema: DatabaseSchema, llm
    ) -> tuple[list[str], list[float | int] | None]:
        """Use the LLM to select relevant tables from a large schema.

        Called when schema.should_use_table_selection() is True (more tables than
        compact_threshold). Passes only table names and descriptions to the LLM,
        then returns the selected table names so full column details for only those
        tables are included in the SQL generation prompt.

        Args:
            state: Conversation state containing the user prompt.
            schema: DatabaseSchema with all available tables.
            llm: LLM instance for table selection.

        Returns:
            List of selected table names. Falls back to all table names if the LLM
            output cannot be parsed (safe degradation).
        """
        compact_schema = schema.get_compact_summary()
        formatted_prompt = Retriever._TABLE_SELECTION_PROMPT.format(
            compact_schema=compact_schema,
            prompt=state.prompt,
        )
        response = llm.invoke(formatted_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        logprobs = llm_tools.extract_logprobs(response)
        raw = raw.strip()

        name_map = {t.name.lower(): t.name for t in schema.tables}
        selected = []
        for token in raw.split(","):
            normalized = token.strip().lower()
            if normalized in name_map:
                selected.append(name_map[normalized])

        if not selected:
            # print("[select_relevant_tables] Warning: could not parse table selection, using all tables")
            selected = [t.name for t in schema.tables]
        return selected, logprobs

    @staticmethod
    def _generate_sql_query(
        state: State, schema_context: str, llm
    ) -> tuple[str, list[float | int] | None]:
        """Generate a DuckDB SQL query from the user prompt and schema context.

        Args:
            state: Conversation state containing the user prompt.
            schema_context: Full schema string produced by DatabaseSchema.get_full_schema_str().
                            Includes table names, descriptions, and column details for all
                            relevant tables.
            llm: LLM instance used to generate the SQL.

        Returns:
            A plain SQL string suitable for DuckDB. Any Markdown fences are stripped.
        """
        formatted_prompt = Retriever._SQL_GENERATION_PROMPT.format(
            prompt=state.prompt,
            schema_context=schema_context,
        )
        response = llm.invoke(formatted_prompt)
        logprobs = llm_tools.extract_logprobs(response)
        sql_query = response.content if hasattr(response, "content") else str(response)
        cleaned_sql = sql_query.strip().replace("```sql", "").replace("```", "")
        return cleaned_sql, logprobs

    def core(self, state: State, llm: BaseChatModel | CoTRefiner) -> State:
        """Core lookup logic - SQL generation and data retrieval.

        Supports both single-table (legacy) and multi-table (new) modes.

        When schema is None, falls back to the legacy behavior of loading DEFAULT_DATA_PATH
        as a single "sales" table, auto-building a minimal DatabaseSchema from it.

        When schema has more tables than compact_threshold, a two-step approach is used:
        first the LLM selects which tables are relevant, then full column details for only
        those tables are passed to SQL generation. This keeps prompts manageable for 10+
        table schemas.

        Args:
            state: Conversation state; must include 'prompt'.
            llm: LLM instance for SQL generation (and optional table selection).

        Returns:
            Updated state containing 'data', 'data_df', 'sql_query' or 'error'.
        """
        schema: DatabaseSchema = self.schema

        # --- Register all tables in a fresh per-call DuckDB connection ---
        con = duckdb.connect()
        for table in schema.tables:
            df_t = pd.read_parquet(table.file_path)
            con.register(f"_df_{table.name}", df_t)
            con.execute(f"CREATE TABLE {table.name} AS SELECT * FROM _df_{table.name}")

        # --- Build schema context (two-step when many tables) ---
        if schema.should_use_table_selection():
            selected_names, logprobs_relevant_tables = (
                Retriever._select_relevant_tables(state, schema, llm)
            )
            schema_context = schema.get_full_schema_str(table_names=selected_names)
        else:
            selected_names = [table.name for table in schema.tables]
            logprobs_relevant_tables = []
            schema_context = schema.get_full_schema_str()

        # --- Generate and execute SQL ---
        sql_query, logprobs_gen_sql = Retriever._generate_sql_query(
            state, schema_context, llm
        )
        try:
            result_df: pd.DataFrame = con.execute(sql_query).df()
            result_str = result_df.to_csv(index=False)

            answer: Answer = Answer(
                agent_id=self.type,
                message=f"The data has been retrieved ({len(result_df)} {'entries' if len(result_df) > 1 else 'entry'} with columns : {', '.join(result_df.columns.to_list())})",
                agent_output={
                    "data_str": result_str,
                    "data_df": result_df,
                    "sql_query": sql_query,
                },
                agent_config=deepcopy(state.get_agent_config(self.type)),
                logprobs=logprobs_relevant_tables + logprobs_gen_sql
                if logprobs_relevant_tables is not None and logprobs_gen_sql is not None
                else None,
            )

            return state.add_answer(answer)

        except Exception as e:
            answer: Answer = Answer(
                agent_id=self.type,
                message="Couldn't access data. Check error message for specific details",
                error=f"Error accessing data: {e!s}",
                agent_config=deepcopy(state.get_agent_config(self.type)),
            )

            return state.add_answer(answer)

    @staticmethod
    def apply_standardization(
        results: list[State], llm: BaseChatModel, original_schema: DatabaseSchema
    ) -> list[State]:

        # Collect candidate info
        candidates = []
        for i, result in enumerate(results):
            last_retriever_answer: Answer | None = result.get_last_answer(
                AgentType.RETRIEVER
            )
            if last_retriever_answer is None:
                continue
            if last_retriever_answer.agent_output["data_df"] is None:
                continue
            df = last_retriever_answer.agent_output["data_df"]
            sql = last_retriever_answer.agent_output["sql_query"]
            if df is None or sql is None:
                continue
            cols = list(df.columns)
            candidates.append(
                {"idx": i, "df": df, "sql": sql, "cols": cols, "state": result}
            )

        if len(candidates) == 0 or len(candidates) == 1:
            return results

        # When all candidates share the same columns AND gt_columns is provided but
        # column names don't already match GT → apply GT alignment directly to all
        # candidates without calling the LLM (no inter-candidate disagreement to resolve).
        col_lists = [tuple(candidate["cols"]) for candidate in candidates]
        if len(set(col_lists)) == 1:  # all lists are equal
            return results
            # column_names_lowered = [c.lower() for c in candidates[0]["cols"]]
            # if column_names_lowered == [c.lower() for c in gt_columns]:
            #     return results
            # # All candidates agree but names don't match GT → rename all without LLM
            # canonical_cols = list(gt_columns)
            # for candidate in candidates:
            #     Retriever._apply_gt_alignment(candidate['df'], canonical_cols)
            # return results

        # Build Prompt
        schema_context = original_schema.get_full_schema_str()
        candidates_lines = []
        for candidate in candidates:
            candidates_lines.append(
                f"Candidate {candidate['idx'] + 1}: SQL: {candidate['sql']} | Columns: {candidate['cols']}"
            )
        candidates_section = "\n".join(candidates_lines)

        # if gt_columns:
        #     gt_hint = (
        #         f"\n## Required Output Column Names (Ground Truth)\n"
        #         f"The canonical_columns in your output MUST be exactly: {gt_columns} (in this order).\n"
        #         f"Rename each candidate column to its semantically matching entry in this list.\n"
        #         f"Do NOT use schema column names or candidate names — use only these GT names."
        #     )
        #     prompt = Retriever._COLUMN_STANDARDIZATION_PROMPT.format(
        #         schema_context=schema_context,
        #         candidates_section=candidates_section,
        #     ) + gt_hint
        # else:

        prompt = Retriever._COLUMN_STANDARDIZATION_PROMPT.format(
            schema_context=schema_context,
            candidates_section=candidates_section,
        )

        # Call LLM
        response = llm.invoke(prompt)
        raw: str = (
            str(response.content) if hasattr(response, "content") else str(response)
        )

        # Parse JSON — strip Markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        mapping_data = json.loads(raw)
        canonical_cols = mapping_data["canonical_columns"]
        mappings = mapping_data["mappings"]

        if len(mappings) != len(candidates):
            return results

        # Apply mappings
        for candidate, col_map in zip(candidates, mappings):
            idx = candidate["idx"]
            state_it: State = results[idx]
            ans_to_check = state_it.get_last_answer(AgentType.RETRIEVER)
            if ans_to_check is None:
                raise AgentException(
                    f"Cannot standardize states with missing {AgentType.RETRIEVER.value} answers"
                )
            ret_ans: Answer = ans_to_check
            if ret_ans.agent_output["data_df"] is None:
                raise AgentException(
                    f"Cannot standardize states with missing {AgentType.RETRIEVER.value} DataFrame"
                )
            df: pd.DataFrame = ret_ans.agent_output["data_df"]

            # Rename
            rename_map = {old: new for old, new in col_map.items() if old in df.columns}
            df = df.rename(columns=rename_map)

            # Reorder to canonical order (only if all canonical cols are present)
            cols_to_order: list[str] = list(canonical_cols)
            if set(cols_to_order).issubset(set(df.columns)):
                df = df.reindex(columns=cols_to_order)

            # Normalize values
            result_df = normalize_dataframe_values(df)

            # Update result
            ret_ans.agent_output["data_df"] = result_df
            ret_ans.agent_output["data_str"] = result_df.to_csv(index=False)
        return results

    def post_generation_hooks(
        self, results: list[State], llm_acc: LLMCallAccumulator, config: AgentConfig
    ) -> list[State]:
        """Use an LLM to standardize column names across best-of-n candidates.

        After best-of-n generates N SQL results, their DataFrames may have different
        column names and orders. This function asks the LLM to determine canonical
        column names and reorders/renames each candidate's DataFrame to match.

        Also applies normalize_dataframe_values to each DataFrame."""

        standardize_llm = llm_tools.get_llm(
            temperature=0.0,
            max_tokens=1000,
            llm_accumulator=llm_acc,
            provider=config.provider,
            model=config.model,
        )

        return Retriever.apply_standardization(
            results, standardize_llm, original_schema=self.schema
        )

    @staticmethod
    def get_evaluator() -> Evaluator:
        return RetrieverEvaluator()
