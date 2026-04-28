from __future__ import annotations

import difflib
import json
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from langchain_ollama import ChatOllama

try:
    from Agent.schema import DatabaseSchema, TableSchema, ColumnSchema
    from Agent.tracing import TracingHelper, _truncate_trace_text, _summarize_dataframe, StatusCode
    from Agent.state import State, DEFAULT_DATA_PATH
except ImportError:
    from schema import DatabaseSchema, TableSchema, ColumnSchema
    from tracing import TracingHelper, _truncate_trace_text, _summarize_dataframe, StatusCode
    from state import State, DEFAULT_DATA_PATH


TABLE_SELECTION_PROMPT = """You are a database architect helping identify which tables are needed to answer a user's question.

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
Example: sales,products
"""


def select_relevant_tables(
    state: "State",
    schema: "DatabaseSchema",
    llm,
    trace_helper: Optional[TracingHelper] = None,
) -> List[str]:
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
    helper = trace_helper or TracingHelper()
    compact_schema = schema.get_compact_summary()
    with helper.start_span(
        "schema_table_selection",
        kind="tool",
        attributes={
            "schema.table_count": len(schema.tables),
            "schema.compact_summary_length": len(compact_schema),
        },
        input_data={
            "prompt": _truncate_trace_text(state.get("prompt", "")),
            "table_count": len(schema.tables),
        },
    ) as span:
        formatted_prompt = TABLE_SELECTION_PROMPT.format(
            compact_schema=compact_schema,
            prompt=state["prompt"],
        )
        response = llm.invoke(formatted_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()

        name_map = {t.name.lower(): t.name for t in schema.tables}
        selected = []
        for token in raw.split(","):
            normalized = token.strip().lower()
            if normalized in name_map:
                selected.append(name_map[normalized])

        if not selected:
            print("[select_relevant_tables] Warning: could not parse table selection, using all tables")
            selected = [t.name for t in schema.tables]
            helper.set_attributes(span, {"selection.fallback_to_all": True})

        helper.set_output(
            span,
            {
                "raw_response": _truncate_trace_text(raw),
                "selected_tables": selected,
            },
        )
        print(f"[select_relevant_tables] Selected tables: {selected}")
        return selected


def _extract_step_output(step_name: str, result: Dict) -> str:
    """Extract the key textual output from a step result for CoT similarity comparison."""
    if step_name == "lookup_sales_data":
        return result.get("sql_query", "")
    elif step_name == "decide_tool":
        return result.get("tool_choice", "")
    else:  # analyzing_data, create_visualization
        answers = result.get("answer", [])
        return answers[-1] if answers else ""


class CoTRefinementLLM:
    """Transparent LLM wrapper that appends the previous iteration's response for iterative CoT refinement.

    Every call to invoke() receives the original prompt augmented with a
    refinement block containing the previous iteration's output.  The wrapper
    delegates all other attribute accesses to the underlying LLM so that the
    core step functions need no changes.
    """

    _REFINEMENT_SUFFIX = """

## ITERATIVE REFINEMENT
Your previous attempt produced the following response:
---
{previous_response}
---
Carefully review your previous response.
- If it is correct and complete, reproduce it exactly (same content, same format).
- If you identify errors or improvements, output a revised version.
Output only the final response with no meta-commentary.
"""

    _ERROR_SUFFIX = """

## ITERATIVE REFINEMENT — EXECUTION ERROR
Your previous attempt produced the following response:
---
{previous_response}
---
When executed, it raised the following error:
---
{execution_error}
---
You MUST fix this error. Output only the corrected response with no meta-commentary.
"""

    def __init__(self, base_llm, previous_response: str, execution_error: str = "") -> None:
        self._llm = base_llm
        self._previous_response = previous_response
        self._execution_error = execution_error

    def invoke(self, prompt):
        if self._execution_error:
            suffix = self._ERROR_SUFFIX.format(
                previous_response=self._previous_response,
                execution_error=self._execution_error,
            )
        else:
            suffix = self._REFINEMENT_SUFFIX.format(
                previous_response=self._previous_response,
            )
        return self._llm.invoke(prompt + suffix)

    def __getattr__(self, name):
        return getattr(self._llm, name)


SQL_GENERATION_PROMPT = """You are an expert SQL developer specializing in DuckDB queries for data analysis and visualization.

## TASK
Generate a DuckDB SQL query to answer the user's question and provide data optimized for visualization.

## AVAILABLE DATA
{schema_context}

## USER QUESTION
{prompt}

## VISUALIZATION GOAL
{visualization_goal}

## INSTRUCTIONS
1. Analyze the user's question to identify what data is needed
2. Consider the visualization goal to structure the query output appropriately
3. Select appropriate columns from the schema above
4. Use proper SQL syntax for filtering, aggregation, sorting, and joins across tables
5. For DATE columns with pattern matching, CAST to VARCHAR: CAST(date_column AS VARCHAR) LIKE '%2021-11%'
6. Handle NULL values appropriately
7. Use DuckDB-specific functions when beneficial
8. **When using JOINs**: always qualify every column reference with its table alias (e.g. `st.region`, not `region`). In SELECT, GROUP BY, ORDER BY, and WHERE, prefix each column with the correct alias of the table it belongs to. Never reference a column by name alone when multiple tables are in scope.

## QUERY OPTIMIZATION FOR VISUALIZATION
- **For time series plots**: Ensure dates are sorted chronologically, use DATE_TRUNC for proper granularity
- **For bar charts**: Aggregate data by category, order by the metric being compared
- **For scatter plots**: Select two numeric columns that show relationships
- **For trend analysis**: Include time-based grouping (daily, monthly, yearly)
- **General**: Limit result size if needed, ensure clean column names for axis labels

## CHAIN OF THOUGHT REASONING
Before generating the SQL query, think step by step:

**Step 1: Understanding the Request**
- What is the user really asking for?
- What is the main entity or metric of interest?
- What time period or filters are implied?

**Step 2: Identifying Required Data**
- Which columns from the schema above are relevant to answer this question?
- Do I need data from multiple tables? If yes, what JOIN keys connect them?
- Do I need to filter the data? If yes, on which column(s)?
- Do I need aggregations (SUM, COUNT, AVG)? If yes, on which column(s)?
- Do I need grouping? If yes, by which column(s)?

**Step 3: Considering Visualization Needs**
- Based on the visualization goal "{visualization_goal}", what chart type is likely?
- For time series: Need chronological ordering and proper date format
- For comparisons: Need categorical grouping and clear labels
- For correlations: Need two numeric columns without aggregation
- What should be on X-axis vs Y-axis?

**Step 4: Query Structure Planning**
- SELECT: Which columns and aggregations?
- FROM: Which table(s)? Use JOINs if data spans multiple tables
- WHERE: What filters are needed?
- GROUP BY: Which columns for aggregation?
- ORDER BY: How should results be sorted?
- LIMIT: Should I limit the result set?

**Step 5: Handling Edge Cases**
- Are there DATE columns that need CAST to VARCHAR for pattern matching?
- Are there potential NULL values that need filtering?
- Do column names need aliasing for better visualization labels?
- Are table aliases needed for clarity in multi-table queries?



## EXAMPLES WITH REASONING

Example 1:
Question: "Show me sales from November 2021"
Visualization: "Monthly sales trend"
Reasoning:
- Step 1: User wants sales data for a specific month
- Step 2: Need Date and Revenue columns from sales table, filter by date pattern
- Step 3: Time series chart → need dates sorted, aggregate by date
- Step 4: SELECT Date, SUM(Revenue), WHERE date matches, GROUP BY Date, ORDER BY Date
- Step 5: Must CAST Date to VARCHAR for LIKE pattern matching
Query: SELECT Date, SUM(Revenue) as Total_Revenue FROM sales WHERE CAST(Date AS VARCHAR) LIKE '%2021-11%' GROUP BY Date ORDER BY Date

Example 2:
Question: "What are the top 5 products by total revenue?"
Visualization: "Compare products by revenue"
Reasoning:
- Step 1: User wants product ranking by revenue
- Step 2: Need Product_Name and Revenue, aggregate revenue per product
- Step 3: Bar chart → categorical comparison, needs ordering, limit to top 5
- Step 4: SELECT Product_Name, SUM(Revenue), GROUP BY product, ORDER BY revenue DESC, LIMIT 5
- Step 5: No special edge cases
Query: SELECT Product_Name, SUM(Revenue) as Total_Revenue FROM sales GROUP BY Product_ID, Product_Name ORDER BY Total_Revenue DESC LIMIT 5

Example 3:
Question: "Show monthly total sales for 2021"
Visualization: "Revenue trends over time"
Reasoning:
- Step 1: User wants monthly aggregation for a specific year
- Step 2: Need Date and Revenue, filter by year, group by month
- Step 3: Time series → need DATE_TRUNC for monthly granularity, chronological order
- Step 4: SELECT DATE_TRUNC('month', Date), SUM(Revenue), WHERE year=2021, GROUP BY month, ORDER BY month
- Step 5: Use EXTRACT for year filtering
Query: SELECT DATE_TRUNC('month', Date) as Month, SUM(Revenue) as Monthly_Sales FROM sales WHERE EXTRACT(YEAR FROM Date) = 2021 GROUP BY Month ORDER BY Month

Example 4:
Question: "Analyze price vs demand relationship"
Visualization: "Price vs demand correlation"
Reasoning:
- Step 1: User wants to see correlation between two variables
- Step 2: Need Price and Units_Sold columns from the same table, no aggregation (scatter plot)
- Step 3: Scatter plot → need individual data points, both axes numeric
- Step 4: SELECT Price, Units_Sold, no GROUP BY needed
- Step 5: Filter out NULLs to avoid chart issues
Query: SELECT Price, Units_Sold FROM sales WHERE Price IS NOT NULL AND Units_Sold IS NOT NULL

Example 5 (multi-table):
Question: "Show total revenue by product category for 2023"
Visualization: "Bar chart of revenue by category"
Schema:
  Table: sales (columns: Sold_Date, SKU_Coded, Total_Sale_Value)
  Table: products (columns: SKU_Coded, Category, Product_Name)
Reasoning:
- Step 1: User wants revenue aggregated by product category
- Step 2: Revenue is in sales; Category is in products — need JOIN on SKU_Coded
- Step 3: Bar chart → group by category, aggregate revenue, order descending
- Step 4: SELECT p.Category, SUM(s.Total_Sale_Value) FROM sales s JOIN products p ON s.SKU_Coded = p.SKU_Coded WHERE year=2023 GROUP BY p.Category ORDER BY revenue DESC
- Step 5: Use EXTRACT for year filter; alias table names for clarity
Query: SELECT p.Category, SUM(s.Total_Sale_Value) as Total_Revenue FROM sales s JOIN products p ON s.SKU_Coded = p.SKU_Coded WHERE EXTRACT(YEAR FROM s.Sold_Date) = 2023 GROUP BY p.Category ORDER BY Total_Revenue DESC

## IMPORTANT — "Average of aggregates" pattern
When the question asks for "average monthly [metric]", "average daily [metric]", etc., you MUST:
1. First aggregate raw rows to the desired period (e.g., compute monthly totals per store using SUM and GROUP BY store + month).
2. Then wrap that result in an outer query or subquery and apply AVG to the aggregated values.
Do NOT apply AVG directly to individual transaction/row values — that gives the average transaction size, not the average monthly metric.

Example 6 (two-level aggregation — "average monthly revenue"):
Question: "Compare average monthly revenue between store regions for 2022 and 2023"
Visualization: "Grouped bar chart comparing average monthly revenue per region between 2022 and 2023"
Schema:
  Table: sales (columns: Sold_Date, Store_Number, Total_Sale_Value)
  Table: stores (columns: Store_Number, region)
Reasoning:
- Step 1: "Average monthly revenue" = compute monthly totals per store first, then average those. NOT AVG(transaction value).
- Step 2: Revenue is in sales; region is in stores — JOIN on Store_Number. Filter years 2022-2023.
- Step 3: Grouped bar → one row per (region, year)
- Step 4: Subquery computes SUM(Total_Sale_Value) per (Store_Number, year, month); outer query JOINs stores and computes AVG(monthly_rev) per (region, year).
- Step 5: Use DATE_TRUNC for monthly grouping; qualify all column references with table aliases.
Query: SELECT st.region, s.yr AS year, ROUND(AVG(s.monthly_rev), 2) AS avg_monthly_revenue FROM (SELECT Store_Number, YEAR(CAST(Sold_Date AS DATE)) AS yr, DATE_TRUNC('month', CAST(Sold_Date AS DATE)) AS month, SUM(Total_Sale_Value) AS monthly_rev FROM sales WHERE YEAR(CAST(Sold_Date AS DATE)) IN (2022, 2023) GROUP BY Store_Number, yr, month) s JOIN stores st ON s.Store_Number = st.Store_Number GROUP BY st.region, s.yr ORDER BY st.region, s.yr

## COMMON MISTAKES TO AVOID
- **NEVER use SUBSTR() or SUBSTRING() directly on a DATE column** — DuckDB DATE columns are not strings.
  WRONG: `WHERE CAST(SUBSTR(Sold_Date, 1, 4) AS INTEGER) = 2021`
  RIGHT: `WHERE YEAR(Sold_Date) = 2021`
- **NEVER use LIKE directly on a DATE column** — cast to VARCHAR first.
  WRONG: `WHERE Sold_Date LIKE '2023%'`
  RIGHT: `WHERE CAST(Sold_Date AS VARCHAR) LIKE '2023%'`
- **NEVER use strptime() on a column that is already DATE type** — it expects a string input.
  WRONG: `CAST(strptime(Sold_Date, '%Y-%m-%d') AS DATE)`
  RIGHT: `Sold_Date` (already DATE, no cast needed)
- **NEVER use strftime(date, format)** — that is SQLite argument order. DuckDB does not support it.
  WRONG: `strftime(Sold_Date, '%Y')`
  RIGHT: `YEAR(Sold_Date)` or `EXTRACT(YEAR FROM Sold_Date)`
- **To extract year from a DATE column**: use `YEAR(date_col)` or `EXTRACT(YEAR FROM date_col)`
- **To extract month from a DATE column**: use `MONTH(date_col)` or `EXTRACT(MONTH FROM date_col)`

## OUTPUT FORMAT
Return ONLY the SQL query as plain text. No explanations. No markdown formatting. No code fences. Just the SQL query.
"""



def generate_sql_query(
    state: State,
    schema_context: str,
    llm,
    trace_helper: Optional[TracingHelper] = None,
) -> str:
    """Generate a DuckDB SQL query from the user prompt and schema context.

    Args:
        state: Conversation state containing the user prompt and optionally visualization_goal.
        schema_context: Full schema string produced by DatabaseSchema.get_full_schema_str().
                        Includes table names, descriptions, and column details for all
                        relevant tables.
        llm: LLM instance used to generate the SQL.

    Returns:
        A plain SQL string suitable for DuckDB. Any markdown fences are stripped.
    """
    visualization_goal = state.get("visualization_goal") or state.get("prompt", "general data analysis")

    helper = trace_helper or TracingHelper()
    with helper.start_span(
        "sql_generation",
        kind="tool",
        attributes={"schema_context_length": len(schema_context)},
        input_data={
            "prompt": _truncate_trace_text(state.get("prompt", "")),
            "visualization_goal": _truncate_trace_text(visualization_goal),
        },
    ) as span:
        formatted_prompt = SQL_GENERATION_PROMPT.format(
            prompt=state["prompt"],
            schema_context=schema_context,
            visualization_goal=visualization_goal,
        )
        response = llm.invoke(formatted_prompt)
        sql_query = response.content if hasattr(response, "content") else str(response)
        cleaned_sql = (
            sql_query.strip()
            .replace("```sql", "")
            .replace("```", "")
        )
        helper.set_output(span, {"sql_query": _truncate_trace_text(cleaned_sql)})
        print("Generated SQL Query:\n", cleaned_sql)
        return cleaned_sql

# -----------------------------
# Core Step Functions (for middleware)
# -----------------------------
# These *_core functions contain just the essential logic without tracing.
# They are called by the middleware for per-step best-of-n execution.

def lookup_sales_data_core(
    state: State,
    llm,
    trace_helper: Optional[TracingHelper] = None,
    *,
    schema: Optional["DatabaseSchema"] = None,
) -> Dict:
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
        schema: DatabaseSchema describing available tables. If None, auto-builds
                a minimal schema from DEFAULT_DATA_PATH for backward compatibility.

    Returns:
        Updated state containing 'data', 'data_df', 'sql_query' or 'error'.
    """
    helper = trace_helper or TracingHelper()

    # --- Build schema if not provided (backward compat) ---
    if schema is None:
        df = pd.read_parquet(DEFAULT_DATA_PATH)
        schema = DatabaseSchema(tables=[TableSchema(
            name="sales",
            description="Sales data",
            file_path=DEFAULT_DATA_PATH,
            columns=[ColumnSchema(name=c, description=c) for c in df.columns.tolist()],
        )])

    # --- Register all tables in a fresh per-call DuckDB connection ---
    con = duckdb.connect()
    for table in schema.tables:
        df_t = pd.read_parquet(table.file_path)
        con.register(f"_df_{table.name}", df_t)
        con.execute(f"CREATE TABLE {table.name} AS SELECT * FROM _df_{table.name}")

    # --- Build schema context (two-step when many tables) ---
    if schema.should_use_table_selection():
        selected_names = select_relevant_tables(state, schema, llm, trace_helper=helper)
        schema_context = schema.get_full_schema_str(table_names=selected_names)
    else:
        selected_names = [table.name for table in schema.tables]
        schema_context = schema.get_full_schema_str()

    # --- Generate and execute SQL ---
    sql_query = generate_sql_query(state, schema_context, llm, trace_helper=helper)
    try:
        with helper.start_span(
            "sql_execution",
            kind="tool",
            attributes={
                "schema_table_count": len(schema.tables),
                "selected_table_count": len(selected_names),
            },
            input_data={"sql_query": _truncate_trace_text(sql_query)},
        ) as span:
            result_df = con.execute(sql_query).df()
            result_str = result_df.to_csv(index=False)
            helper.set_output(
                span,
                {
                    "selected_tables": selected_names,
                    "dataframe": _summarize_dataframe(result_df),
                    "data_preview": _truncate_trace_text(result_str),
                },
            )
        return {**state, "data": result_str, "data_df": result_df, "sql_query": sql_query}
    except Exception as e:
        print(f"Error accessing data: {str(e)}")
        return {**state, "data": "", "sql_query": sql_query, "error": f"Error accessing data: {str(e)}"}


def _enrich_data_with_stats(data_csv: str) -> str:
    """Append pre-computed numeric statistics to the CSV data string.

    LLMs are unreliable at mental arithmetic over many rows.  Pre-computing
    sum / min / max / count for every numeric column and appending them as a
    summary block lets the LLM read the answer directly instead of deriving it.
    """
    if not data_csv or not data_csv.strip():
        return data_csv
    try:
        import io
        df = pd.read_csv(io.StringIO(data_csv))
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return data_csv
        lines = ["\n--- Pre-computed Statistics (use these exact values) ---"]
        lines.append(f"Total rows: {len(df)}")
        for col in num_cols:
            s = df[col]
            lines.append(
                f"{col}: sum={round(s.sum(), 2)}, min={round(s.min(), 2)}, "
                f"max={round(s.max(), 2)}, mean={round(s.mean(), 2)}"
            )
        return data_csv + "\n".join(lines)
    except Exception:
        return data_csv


def analyzing_data_core(state: State, llm, trace_helper: Optional[TracingHelper] = None) -> Dict:
    """Core analysis logic - LLM-based data analysis.

    Args:
        state: Conversation state; should include 'data' and 'prompt'.
        llm: LLM instance for analysis.

    Returns:
        Updated state with analysis appended to 'answer'.
    """
    try:
        enriched_data = _enrich_data_with_stats(state.get("data", ""))
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(
            data=enriched_data, prompt=state.get("prompt", ""), sql_query=state.get("sql_query", "")
        )
        analysis_result = llm.invoke(formatted_prompt)
        analysis_text = analysis_result.content if hasattr(analysis_result, "content") else str(analysis_result)
        print(f"Analysis:\n{analysis_text}")
        return {
            **state,
            "answer": state.get("answer", []) + [analysis_text],
        }
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


def decide_tool_core(state: State, llm, trace_helper: Optional[TracingHelper] = None) -> Dict:
    """Core tool decision logic - LLM-based routing.

    Args:
        state: Conversation state.
        llm: LLM instance for decision.

    Returns:
        Updated state with 'tool_choice'.
    """
    tools_description = """You are a workflow orchestrator managing a data analysis pipeline.

## AVAILABLE TOOLS
- lookup_sales_data: Retrieves data from the database using SQL
- analyzing_data: Analyzes retrieved data and provides insights
- create_visualization: Generates chart code to visualize the data
- end: Completes the workflow

## DECISION RULES (CRITICAL - Follow in order)
1. Data prerequisite: Must run lookup_sales_data BEFORE analyzing_data or create_visualization
2. No repetition: NEVER select a tool that has already been used
3. Completion criteria: Select 'end' when:
   - 2 or more answers have been generated (analysis + visualization complete)
   - All relevant tools for the user's request have been executed

## DECISION FLOWCHART
Start → Has data? No → lookup_sales_data
              ↓ Yes
          Already analyzed? No → analyzing_data
              ↓ Yes
          Need visualization? Yes → create_visualization
              ↓ No/Done
          end
    """

    decision_prompt = f"""
    {tools_description}

## CURRENT STATE
- User's request: {state.get('prompt')}
- Answers generated so far: {state.get('answer', [])}
- Visualization goal: {state.get('visualization_goal')}
- Last tool used: {state.get('tool_choice')}

## CHAIN OF THOUGHT REASONING
Before selecting the next tool, think step by step:

**Step 1: Analyzing User Request**
- What is the user asking for? (data lookup, analysis, visualization, or combination)
- Does the request explicitly or implicitly require a chart/graph?
- Is this a simple data retrieval or complex multi-step task?

**Step 2: Checking Current Progress**
- What tools have already been executed? (check Last tool used)
- Do we have data available? (check if lookup_sales_data was run)
- How many answers have been generated? (check Answers generated so far)
- What stage of the workflow are we in?

**Step 3: Identifying What's Missing**
- If no data: Need lookup_sales_data first (Rule 1)
- If data exists but no analysis: Need analyzing_data
- If analysis exists but user wants visualization: Need create_visualization
- If all required steps done: Need end

**Step 4: Applying Decision Rules**
- Rule 1 check: Do I have data before attempting analysis/visualization?
- Rule 2 check: Am I about to repeat a tool already used?
- Rule 3 check: Have I completed all necessary steps (2+ answers OR all relevant tools)?

**Step 5: Making the Decision**
- Based on steps 1-4, which tool should execute next?
- Does this choice follow the DECISION FLOWCHART?
- Is this the minimum necessary step to progress toward completion?

## EXAMPLES WITH REASONING

Example 1 - Initial state:
State: prompt="Show sales data", answer=[], tool_choice=None

Reasoning:
- Step 1: User wants sales data (implies lookup needed)
- Step 2: No tools executed yet, no data, no answers
- Step 3: Missing everything - start with data retrieval
- Step 4: Rule 1 applies - need data first, Rule 2 N/A (nothing used), Rule 3 not met (0 answers)
- Step 5: Must start with lookup_sales_data

Decision: lookup_sales_data (need data first)

Example 2 - After data lookup:
State: prompt="Show sales data", answer=[], tool_choice="lookup_sales_data", data exists

Reasoning:
- Step 1: User wants sales data shown (implies analysis/presentation needed)
- Step 2: lookup_sales_data executed, data available, but 0 answers generated
- Step 3: Have data, missing analysis
- Step 4: Rule 1 satisfied (have data), Rule 2 check (can't repeat lookup), Rule 3 not met (0 answers)
- Step 5: Next logical step is analyzing_data

Decision: analyzing_data (have data, now analyze)

Example 3 - After analysis and visualization:
State: prompt="Show sales trends", answer=["Analysis text", "Chart code"], tool_choice="create_visualization"

Reasoning:
- Step 1: User wanted trends (implies analysis + visualization)
- Step 2: All tools executed, 2 answers generated (analysis + chart code)
- Step 3: Nothing missing - workflow complete
- Step 4: Rule 1 satisfied, Rule 2 satisfied, Rule 3 MET (2+ answers generated)
- Step 5: Should end the workflow

Decision: end (2+ answers generated, workflow complete)

Example 4 - After analysis only (no viz needed):
State: prompt="What were total sales?", answer=["Total sales were $X"], tool_choice="analyzing_data"

Reasoning:
- Step 1: User wanted a simple factual answer (no visualization implied)
- Step 2: lookup and analysis executed, 1 answer generated
- Step 3: Question fully answered with analysis alone
- Step 4: Rule 1 satisfied, Rule 2 satisfied, Rule 3 check (all RELEVANT tools done)
- Step 5: No visualization needed for this query - can end

Decision: end (all relevant tools executed, question answered)

Example 5 - After lookup only (viz needed):
State: prompt="Show me a chart of monthly sales", answer=[], tool_choice="lookup_sales_data", data exists

Reasoning:
- Step 1: User explicitly wants a chart (visualization required)
- Step 2: Only lookup executed, data available, 0 answers
- Step 3: Missing both analysis AND visualization
- Step 4: Rule 1 satisfied (have data), Rule 2 satisfied (not repeating), Rule 3 not met (0 answers)
- Step 5: Should analyze first, then visualize (follow flowchart)

Decision: analyzing_data (analyze before visualizing)

## YOUR TASK
Based on the chain of thought reasoning above and the current state, select the next tool to execute.

## OUTPUT FORMAT
Respond with ONLY the tool name: lookup_sales_data, analyzing_data, create_visualization, or end
No explanations. Just the tool name.
    """

    try:
        current_prompt = state.get("prompt", "")
        current_answer = state.get("answer", [])
        visualization_goal = state.get("visualization_goal")
        chart_config = state.get("chart_config")

        response = llm.invoke(decision_prompt)
        tool_choice = response.content.strip().lower()
        valid_tools = ["lookup_sales_data", "analyzing_data", "create_visualization", "end"]
        closest_match = difflib.get_close_matches(tool_choice, valid_tools, n=1, cutoff=0.6)
        matched_tool = closest_match[0] if closest_match else "lookup_sales_data"

        if matched_tool in ["analyzing_data", "create_visualization"] and not state.get("data"):
            matched_tool = "lookup_sales_data"
        # Anti-loop guard: if lookup already ran but returned no data (SQL error), stop
        if matched_tool == "lookup_sales_data" and state.get("tool_choice") == "lookup_sales_data" and not state.get("data"):
            print("[decide_tool] lookup_sales_data already ran but returned no data — forcing end to avoid infinite loop")
            matched_tool = "end"
        elif len(state.get("answer", [])) > 1:
            matched_tool = "end"

        print(f"\n\nTool selected: {matched_tool}")

        return {
            **state,
            "prompt": current_prompt,
            "answer": current_answer,
            "visualization_goal": visualization_goal,
            "chart_config": chart_config,
            "tool_choice": matched_tool,
        }
    except Exception as e:
        print(f"Error deciding tool: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


def create_visualization_core(state: State, llm, trace_helper: Optional[TracingHelper] = None) -> Dict:
    """Core visualization logic - chart config extraction and code generation.

    Args:
        state: Conversation state; should include 'data_df' (DataFrame).
        llm: LLM instance for config extraction and code generation.

    Returns:
        Updated state with 'chart_config' and code appended to 'answer'.
        If the generated code raises an exception when executed, the result
        also contains an 'error' key so the CoT refinement loop can feed the
        error message back to the LLM on the next iteration.
    """
    try:
        data_df = state.get("data_df")
        helper = trace_helper or TracingHelper()

        if data_df is not None:
            print(f"Using DataFrame with shape: {data_df.shape}, columns: {list(data_df.columns)}")
        else:
            print("Warning: No DataFrame available in state")

        # Extract chart configuration
        with_config = extract_chart_config(state, llm, trace_helper=helper)

        # Ensure DataFrame is in the updated state
        with_config["data_df"] = data_df

        # Generate chart code
        code = create_chart(with_config, llm, trace_helper=helper)

        # --- Validate by executing in a headless namespace (no display) ---
        # Switch to Agg (non-interactive) backend to avoid tkinter threading
        # issues when running best-of-n from a non-main thread on Windows.
        exec_code = (
            "import matplotlib.pyplot as plt; plt.switch_backend('Agg')\n"
            + code.replace("plt.show()", "plt.close('all')")
        )
        namespace: Dict = {
            "data_df": data_df,
            "config": with_config.get("chart_config", {}),
        }
        try:
            with helper.start_span(
                "visualization_validation",
                kind="tool",
                input_data={
                    "chart_config": with_config.get("chart_config", {}),
                    "dataframe": _summarize_dataframe(data_df),
                    "code": _truncate_trace_text(code),
                },
            ) as span:
                exec(exec_code, namespace)  # noqa: S102
                exec_error = ""
                helper.set_output(span, {"validation": "passed"})
        except Exception as e:
            exec_error = f"{type(e).__name__}: {e}"
            print(f"[create_visualization] Code validation error: {exec_error}")

        result: Dict = {
            **with_config,
            "answer": with_config.get("answer", []) + [code],
        }
        if exec_error:
            result["error"] = exec_error
        return result
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


# -----------------------------
# Original Step Functions (with tracing support)
# -----------------------------

def lookup_sales_data(state: State, llm, tracer=None, *, schema: Optional["DatabaseSchema"] = None) -> Dict:
    """Look up data using LLM-generated SQL over DuckDB.

    Delegates to lookup_sales_data_core for the core logic, then wraps the result
    in a tracing span if a tracer is provided.

    Args:
        state: Conversation state; must include 'prompt'.
        llm: LLM instance used for prompt-to-SQL generation.
        tracer: Optional Phoenix/OpenInference tracer for observability.
        schema: DatabaseSchema describing available tables. If None, falls back to
                loading DEFAULT_DATA_PATH as a single "sales" table.

    Returns:
        Updated state containing 'data' (string table), 'data_df', 'sql_query', or 'error'.
    """
    result = lookup_sales_data_core(state, llm, schema=schema)
    if tracer is not None:
        try:
            result_str = result.get("data", "")
            with tracer.start_as_current_span("sql_query_exec", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                span.set_input(state.get("prompt", ""))  # type: ignore[attr-defined]
                span.set_output(result_str)  # type: ignore[attr-defined]
                if StatusCode is not None:
                    span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
        except Exception:
            pass
    return result

DATA_ANALYSIS_PROMPT = """You are a professional data analyst providing insights from query results.

## TASK
Answer the user's question based ONLY on the provided data.

## USER QUESTION
{prompt}

## AVAILABLE DATA
This data was retrieved using the SQL query: {sql_query}

Data:
{data}

## INSTRUCTIONS
1. Examine the data carefully to understand what information is available
2. Identify the key insights that directly answer the user's question
3. Provide a concise, specific answer (2-3 sentences maximum)
4. Use actual numbers and facts from the data
5. Do NOT speculate or make assumptions beyond what the data shows
6. If the data doesn't fully answer the question, state what you can determine from the available data

## CHAIN OF THOUGHT REASONING
Before answering, think step by step:

**Step 1: Understanding the Question**
- What specific information is the user asking for?
- Is it asking for a single value, a comparison, a trend, or a summary?
- What would constitute a complete answer?

**Step 2: Examining the Data Structure**
- How many rows of data are available?
- What columns are present in the data?
- What is the range or distribution of values?
- Are there any patterns or anomalies visible?

**Step 3: Extracting Relevant Facts**
- Which specific values directly answer the question?
- Do I need to perform mental calculations (sum, average, count)?
- What are the exact numbers, dates, or categories relevant to the answer?
- Are there any context clues (time periods, units, categories)?

**Step 4: Verifying Completeness**
- Does the data fully answer the user's question?
- Is there missing information that prevents a complete answer?
- Should I mention any limitations or caveats?

**Step 5: Formulating the Answer**
- How can I state the facts concisely (2-3 sentences)?
- Am I using specific numbers from the data?
- Am I avoiding speculation or assumptions?
- Is my answer direct and clear?



## EXAMPLES WITH REASONING

Example 1 - Good answer:
Question: "What were the total sales in November 2021?"
Data: Shows 45 rows with Revenue column summing to $1,234,567

Reasoning:
- Step 1: User wants total sales amount for a specific month
- Step 2: 45 rows of data, Revenue column present
- Step 3: Sum of Revenue = $1,234,567, time period = November 2021, transaction count = 45
- Step 4: Data fully answers the question, no missing info
- Step 5: State the total, mention the number of transactions, keep it factual

Answer: "Based on the data, total sales in November 2021 were $1,234,567 across 45 transactions."

Example 2 - Bad answer (do NOT do this):
Question: "What were the total sales in November 2021?"
Data: Shows 45 rows with Revenue column summing to $1,234,567

Bad Answer: "Sales were strong in November, likely due to holiday shopping. This trend probably continued into December and suggests the company is performing well."

Why this is bad:
- Violates Step 5: Adds speculation ("likely due to holiday shopping")
- Violates instruction 5: Makes assumptions beyond data ("trend continued")
- Violates Step 3: Doesn't state the actual number ($1,234,567)
- Adds interpretation not supported by data ("company performing well")

Example 3 - Handling incomplete data:
Question: "How do our November 2021 sales compare to the previous year?"
Data: Shows only November 2021 data (45 rows, $1,234,567 total)

Reasoning:
- Step 1: User wants year-over-year comparison
- Step 2: Only 2021 data present, no 2020 data
- Step 3: Can extract November 2021 total = $1,234,567
- Step 4: Cannot make comparison - missing 2020 data
- Step 5: State what we know, acknowledge limitation

Answer: "The available data shows November 2021 sales totaled $1,234,567 across 45 transactions. However, the dataset does not include November 2020 data, so a year-over-year comparison cannot be made."

Example 4 - Multiple data points:
Question: "Which product had the highest revenue?"
Data: Shows Product_Name and Revenue for 10 products, top one is "Widget Pro" with $450,000

Reasoning:
- Step 1: User wants to identify top-performing product
- Step 2: 10 rows, columns are Product_Name and Revenue
- Step 3: Maximum Revenue = $450,000, corresponding Product_Name = "Widget Pro"
- Step 4: Data fully answers the question
- Step 5: State the product name and its revenue value

Answer: "Widget Pro had the highest revenue at $450,000."

## OUTPUT FORMAT
Provide a direct, concise answer in natural language (2-3 sentences). Focus only on facts from the data.
"""

def analyzing_data(state: State, llm: ChatOllama, tracer=None) -> Dict:
    """Ask the LLM to analyze the looked-up data in the context of the prompt.

    Args:
        state: Conversation state; should include 'data' and 'prompt'.
        llm: ChatOllama instance used for the analysis.

    Returns:
        Updated state including the analysis appended to 'answer'.
    """
    try:
        #print("Data to analyze:\n", state.get("data", ""))
        enriched_data = _enrich_data_with_stats(state.get("data", ""))
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(
            data=enriched_data, prompt=state.get("prompt", ""), sql_query=state.get("sql_query","")
        )
        analysis_result = llm.invoke(formatted_prompt)
        analysis_text = analysis_result.content if hasattr(analysis_result, "content") else str(analysis_result)
        if tracer is not None:
            try:
                with tracer.start_as_current_span("data_analysis", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                    span.set_input(state.get("prompt", ""))  # type: ignore[attr-defined]
                    span.set_output(str(analysis_text))  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
            except Exception:
                pass
        return {
            **state,
            "answer": state.get("answer", []) + [analysis_text],
        }
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}

def decide_tool(state: State, llm: ChatOllama, tracer=None) -> State:
    """Select the next tool to run given the current conversation state.

    The LLM is prompted with the available tools and minimal state. The raw
    response is normalized against a fixed list of valid tool names.

    Tool selection constraints:
    - If no data is present, force 'lookup_sales_data' before analysis/visualization.
    - If more than one answer message is present, end the flow ('end').

    Args:
        state: Conversation state.
        llm: ChatOllama instance used to decide the tool.

    Returns:
        Updated state including 'tool_choice'.
    """
    tools_description = """You are a workflow orchestrator managing a data analysis pipeline.

## AVAILABLE TOOLS
- lookup_sales_data: Retrieves data from the database using SQL
- analyzing_data: Analyzes retrieved data and provides insights
- create_visualization: Generates chart code to visualize the data
- end: Completes the workflow

## DECISION RULES (CRITICAL - Follow in order)
1. Data prerequisite: Must run lookup_sales_data BEFORE analyzing_data or create_visualization
2. No repetition: NEVER select a tool that has already been used
3. Completion criteria: Select 'end' when:
   - 2 or more answers have been generated (analysis + visualization complete)
   - All relevant tools for the user's request have been executed

## DECISION FLOWCHART
Start → Has data? No → lookup_sales_data
              ↓ Yes
          Already analyzed? No → analyzing_data
              ↓ Yes
          Need visualization? Yes → create_visualization
              ↓ No/Done
          end
    """

    decision_prompt = f"""
    {tools_description}

## CURRENT STATE
- User's request: {state.get('prompt')}
- Answers generated so far: {state.get('answer', [])}
- Visualization goal: {state.get('visualization_goal')}
- Last tool used: {state.get('tool_choice')}

## CHAIN OF THOUGHT REASONING
Before selecting the next tool, think step by step:

**Step 1: Analyzing User Request**
- What is the user asking for? (data lookup, analysis, visualization, or combination)
- Does the request explicitly or implicitly require a chart/graph?
- Is this a simple data retrieval or complex multi-step task?

**Step 2: Checking Current Progress**
- What tools have already been executed? (check Last tool used)
- Do we have data available? (check if lookup_sales_data was run)
- How many answers have been generated? (check Answers generated so far)
- What stage of the workflow are we in?

**Step 3: Identifying What's Missing**
- If no data: Need lookup_sales_data first (Rule 1)
- If data exists but no analysis: Need analyzing_data
- If analysis exists but user wants visualization: Need create_visualization
- If all required steps done: Need end

**Step 4: Applying Decision Rules**
- Rule 1 check: Do I have data before attempting analysis/visualization?
- Rule 2 check: Am I about to repeat a tool already used?
- Rule 3 check: Have I completed all necessary steps (2+ answers OR all relevant tools)?

**Step 5: Making the Decision**
- Based on steps 1-4, which tool should execute next?
- Does this choice follow the DECISION FLOWCHART?
- Is this the minimum necessary step to progress toward completion?


## EXAMPLES WITH REASONING

Example 1 - Initial state:
State: prompt="Show sales data", answer=[], tool_choice=None

Reasoning:
- Step 1: User wants sales data (implies lookup needed)
- Step 2: No tools executed yet, no data, no answers
- Step 3: Missing everything - start with data retrieval
- Step 4: Rule 1 applies - need data first, Rule 2 N/A (nothing used), Rule 3 not met (0 answers)
- Step 5: Must start with lookup_sales_data

Decision: lookup_sales_data (need data first)

Example 2 - After data lookup:
State: prompt="Show sales data", answer=[], tool_choice="lookup_sales_data", data exists

Reasoning:
- Step 1: User wants sales data shown (implies analysis/presentation needed)
- Step 2: lookup_sales_data executed, data available, but 0 answers generated
- Step 3: Have data, missing analysis
- Step 4: Rule 1 satisfied (have data), Rule 2 check (can't repeat lookup), Rule 3 not met (0 answers)
- Step 5: Next logical step is analyzing_data

Decision: analyzing_data (have data, now analyze)

Example 3 - After analysis and visualization:
State: prompt="Show sales trends", answer=["Analysis text", "Chart code"], tool_choice="create_visualization"

Reasoning:
- Step 1: User wanted trends (implies analysis + visualization)
- Step 2: All tools executed, 2 answers generated (analysis + chart code)
- Step 3: Nothing missing - workflow complete
- Step 4: Rule 1 satisfied, Rule 2 satisfied, Rule 3 MET (2+ answers generated)
- Step 5: Should end the workflow

Decision: end (2+ answers generated, workflow complete)

Example 4 - After analysis only (no viz needed):
State: prompt="What were total sales?", answer=["Total sales were $X"], tool_choice="analyzing_data"

Reasoning:
- Step 1: User wanted a simple factual answer (no visualization implied)
- Step 2: lookup and analysis executed, 1 answer generated
- Step 3: Question fully answered with analysis alone
- Step 4: Rule 1 satisfied, Rule 2 satisfied, Rule 3 check (all RELEVANT tools done)
- Step 5: No visualization needed for this query - can end

Decision: end (all relevant tools executed, question answered)

Example 5 - After lookup only (viz needed):
State: prompt="Show me a chart of monthly sales", answer=[], tool_choice="lookup_sales_data", data exists

Reasoning:
- Step 1: User explicitly wants a chart (visualization required)
- Step 2: Only lookup executed, data available, 0 answers
- Step 3: Missing both analysis AND visualization
- Step 4: Rule 1 satisfied (have data), Rule 2 satisfied (not repeating), Rule 3 not met (0 answers)
- Step 5: Should analyze first, then visualize (follow flowchart)

Decision: analyzing_data (analyze before visualizing)

## YOUR TASK
Based on the chain of thought reasoning above and the current state, select the next tool to execute.


## OUTPUT FORMAT
Respond with ONLY the tool name: lookup_sales_data, analyzing_data, create_visualization, or end
No explanations. Just the tool name.
    """

    try:
        current_prompt = state.get("prompt", "")
        current_answer = state.get("answer", [])
        visualization_goal = state.get("visualization_goal")
        chart_config = state.get("chart_config")

        response = llm.invoke(decision_prompt)
        tool_choice = response.content.strip().lower()
        valid_tools = ["lookup_sales_data", "analyzing_data", "create_visualization", "end"]
        closest_match = difflib.get_close_matches(tool_choice, valid_tools, n=1, cutoff=0.6)
        matched_tool = closest_match[0] if closest_match else "lookup_sales_data"

        if matched_tool in ["analyzing_data", "create_visualization"] and not state.get("data"):
            matched_tool = "lookup_sales_data"
        # Anti-loop guard: if lookup already ran but returned no data (SQL error), stop
        if matched_tool == "lookup_sales_data" and state.get("tool_choice") == "lookup_sales_data" and not state.get("data"):
            print("[decide_tool] lookup_sales_data already ran but returned no data — forcing end to avoid infinite loop")
            matched_tool = "end"
        elif len(state.get("answer", [])) > 1:
            matched_tool = "end"

        print(f"\n\nTool selected: {matched_tool}")

        return {
            **state,
            "prompt": current_prompt,
            "answer": current_answer,
            "visualization_goal": visualization_goal,
            "chart_config": chart_config,
            "tool_choice": matched_tool,
        }
    except Exception as e:
        print(f"Error deciding tool: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


CHART_CONFIGURATION_PROMPT = """You are a data visualization expert designing chart configurations.

## TASK
Create a JSON configuration object for visualizing the provided data.

## VISUALIZATION GOAL
{visualization_goal}

## DATA TO VISUALIZE
{data}

## CHART TYPE SELECTION GUIDE
Choose the appropriate chart type based on the data and goal:
- bar: Comparing discrete categories or groups (e.g., sales by product, revenue by region)
- line: Showing trends over time or continuous progression (e.g., monthly sales, daily visitors)
- scatter: Showing correlations or relationships between two variables (e.g., price vs. demand)
- area: Showing volume or cumulative values over time (e.g., cumulative revenue, market share)

## REQUIRED JSON KEYS
- chart_type: One of [bar, line, area, scatter]
- x_axis: Column name for X-axis (string)
- y_axis: Column name for Y-axis (string) — use this for SINGLE-series charts and for long-format grouped bar charts (used together with group_by)
- y_axes: List of column names for Y-axis (list of strings) — use this INSTEAD of y_axis when the DataFrame already has one column per series (wide format). Do NOT include both y_axis and y_axes.
- group_by: (OPTIONAL) Column name whose distinct values define the bar series in a long-format grouped bar chart. Use this together with y_axis when the data has a discriminator column (e.g., 'year', 'quarter') instead of separate columns per series. Do NOT use together with y_axes.
- title: Descriptive chart title (string)

## WHEN TO USE y_axes vs y_axis vs group_by
- Use y_axis (single string) when showing ONE metric: revenue, count, score
- Use y_axes (list) when the DataFrame already has one column per series — i.e., the series values are in separate columns (e.g., Avg_Revenue_Promo, Avg_Revenue_Non_Promo)
- Use y_axis + group_by when data is in LONG FORMAT with a discriminator column: the same metric column (y_axis) appears for multiple groups identified by another column (group_by). Example: data has columns (region, year, avg_monthly_revenue) and you want separate bars for each year → x_axis=region, y_axis=avg_monthly_revenue, group_by=year

## CHAIN OF THOUGHT REASONING
Before creating the configuration, think step by step:

**Step 1: Understanding the Visualization Goal**
- What story does the user want to tell with this chart?
- Is the goal to compare, show trends, find correlations, or display distributions?
- Are there any keywords that hint at chart type? (e.g., "over time" → line, "compare" → bar)

**Step 2: Analyzing the Data Structure**
- What columns are available in the data?
- Which columns contain categorical data (text, discrete values)?
- Which columns contain numerical data (integers, floats)?
- Are there any date/time columns?
- How many rows of data are there (affects visualization approach)?

**Step 3: Selecting Chart Type**
- For comparisons between categories → bar chart
- For trends over time or continuous sequences → line chart
- For showing relationships between two numeric variables → scatter plot
- For cumulative values over time → area chart
- Does the data structure support this chart type?

**Step 4: Mapping Axes**
- What should go on the X-axis? (independent variable, categories, or time)
- What should go on the Y-axis? (dependent variable, values, metrics)
- Do the chosen columns make logical sense for these axes?
- For time series: dates on X-axis, metrics on Y-axis
- For comparisons: categories on X-axis, values on Y-axis

**Step 5: Creating the Title**
- What concise phrase describes what the chart shows?
- Include the key variables or metrics being displayed
- Format: "[Metric] by/over/vs [Variable]" or similar clear structure
- Examples: "Revenue Over Time", "Sales by Product", "Price vs Demand"

Now, based on this reasoning, create the JSON configuration.

## EXAMPLES WITH REASONING

Example 1 - Time series data:
Data columns: Date, Revenue
Goal: "Show revenue trends over time"

Reasoning:
- Step 1: Goal mentions "trends over time" → time series visualization
- Step 2: Columns: Date (temporal), Revenue (numeric), likely multiple rows
- Step 3: "Over time" + "trends" → line chart is appropriate
- Step 4: X-axis = Date (time progression), Y-axis = Revenue (metric being tracked)
- Step 5: Title: "Revenue Trends Over Time" (clear, includes both variables)

Output: {{"chart_type": "line", "x_axis": "Date", "y_axis": "Revenue", "title": "Revenue Trends Over Time"}}

Example 2 - Categorical comparison:
Data columns: Product_Name, Units_Sold
Goal: "Compare products by units sold"

Reasoning:
- Step 1: Goal says "compare" → comparison visualization
- Step 2: Columns: Product_Name (categorical), Units_Sold (numeric)
- Step 3: Comparing discrete categories → bar chart is appropriate
- Step 4: X-axis = Product_Name (categories), Y-axis = Units_Sold (values to compare)
- Step 5: Title: "Units Sold by Product" (shows what's being compared)

Output: {{"chart_type": "bar", "x_axis": "Product_Name", "y_axis": "Units_Sold", "title": "Units Sold by Product"}}

Example 3 - Correlation analysis:
Data columns: Price, Demand, Product_ID
Goal: "Analyze the relationship between price and demand"

Reasoning:
- Step 1: Goal mentions "relationship between" → correlation visualization
- Step 2: Columns: Price (numeric), Demand (numeric), Product_ID (identifier)
- Step 3: Two numeric variables, looking for correlation → scatter plot
- Step 4: X-axis = Price (independent variable), Y-axis = Demand (dependent variable)
- Step 5: Title: "Price vs Demand Analysis" (shows both variables being correlated)

Output: {{"chart_type": "scatter", "x_axis": "Price", "y_axis": "Demand", "title": "Price vs Demand Analysis"}}

Example 4 - Cumulative values:
Data columns: Month, Cumulative_Sales
Goal: "Show cumulative sales growth throughout the year"

Reasoning:
- Step 1: Goal mentions "cumulative" and "growth" → volume over time
- Step 2: Columns: Month (temporal), Cumulative_Sales (numeric, accumulating)
- Step 3: Cumulative values over time → area chart emphasizes volume
- Step 4: X-axis = Month (time), Y-axis = Cumulative_Sales (accumulated metric)
- Step 5: Title: "Cumulative Sales Growth" (describes the accumulation)

Output: {{"chart_type": "area", "x_axis": "Month", "y_axis": "Cumulative_Sales", "title": "Cumulative Sales Growth"}}

Example 5 - Regional comparison:
Data columns: Region, Average_Revenue, Store_Count
Goal: "Compare average revenue across different regions"

Reasoning:
- Step 1: Goal says "compare...across" → categorical comparison
- Step 2: Columns: Region (categorical), Average_Revenue (numeric), Store_Count (numeric)
- Step 3: Comparing categories → bar chart
- Step 4: X-axis = Region (categories), Y-axis = Average_Revenue (metric from goal)
- Step 5: Title: "Average Revenue by Region" (clear comparison statement)

Output: {{"chart_type": "bar", "x_axis": "Region", "y_axis": "Average_Revenue", "title": "Average Revenue by Region"}}

Example 6 - Multi-series comparison (grouped bar):
Data columns: Product_Class, Avg_Revenue_Promo, Avg_Revenue_Non_Promo, Abs_Difference
Goal: "Compare average revenue per unit during promotions vs non-promotions for each product class"

Reasoning:
- Step 1: Goal says "compare...vs" → TWO metrics side by side, not one
- Step 2: Columns: Product_Class (categorical), Avg_Revenue_Promo and Avg_Revenue_Non_Promo (both numeric, both needed)
- Step 3: Comparing two numeric series across categories → grouped bar chart
- Step 4: X-axis = Product_Class (categories), Y-axes = [Avg_Revenue_Promo, Avg_Revenue_Non_Promo] (both series)
- Step 5: Title clearly names both series being compared

Output: {{"chart_type": "bar", "x_axis": "Product_Class", "y_axes": ["Avg_Revenue_Promo", "Avg_Revenue_Non_Promo"], "title": "Average Revenue per Unit: Promo vs Non-Promo by Product Class"}}

Example 7 - Long-format grouped bar chart (data has a discriminator column, use group_by):
Data columns: region, year, avg_monthly_revenue (8 rows: 4 regions × 2 years, long format)
Goal: "Compare average monthly revenue by region for 2022 vs 2023"

Reasoning:
- Step 1: "Compare...for 2022 vs 2023" → two bar series, one per year
- Step 2: Columns: region (categorical x-axis), year (discriminator: values 2022 or 2023), avg_monthly_revenue (metric). Data is LONG FORMAT — one row per (region, year). There are NO separate columns avg_monthly_revenue_2022 / avg_monthly_revenue_2023.
- Step 3: Two series over categories → grouped bar chart
- Step 4: x_axis = region, y_axis = avg_monthly_revenue (the metric), group_by = year (to split into two bar series). Do NOT use y_axes here — that would require wide-format columns which don't exist.
- Step 5: Title names both variables being compared

Output: {{"chart_type": "bar", "x_axis": "region", "y_axis": "avg_monthly_revenue", "group_by": "year", "title": "Avg Monthly Revenue by Region: 2022 vs 2023"}}


## OUTPUT FORMAT
Return ONLY a valid JSON object. No markdown. No code fences. No backticks. No explanations. Just the JSON.
"""


def _parse_chart_config(raw_text: str) -> Dict[str, str]:
    """Parse a chart configuration JSON from a raw LLM response.

    The function attempts to tolerate code fences and extra prose, extracting the
    first JSON object it can find. On failure, a minimal default schema is
    returned.

    Args:
        raw_text: Raw text from the LLM expected to contain a JSON object.

    Returns:
        A dictionary with keys: 'chart_type', 'x_axis', 'y_axis', 'title'.
    """
    text = raw_text.strip().strip("`")
    # Attempt to extract JSON from possible code fences or prose
    try:
        # If there's a fenced block like ```json ... ``` remove it
        if text.lower().startswith("json"):  # e.g., "json\n{...}"
            text = text[4:].strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        # Try to find first JSON object in text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass
    # Fallback minimal schema
    return {
        "chart_type": "line",
        "x_axis": "date",
        "y_axis": "value",
        "title": "Chart",
    }


def extract_chart_config(
    state: State,
    llm: ChatOllama,
    trace_helper: Optional[TracingHelper] = None,
) -> State:
    """Infer a compact chart configuration from the looked-up data.

    Prompts the LLM to return a minified JSON config and parses it into a
    Python dict. Data is NOT included in the config (it's passed separately as DataFrame).

    Args:
        state: Conversation state; should include 'data' and optionally 'visualization_goal'.
        llm: ChatOllama instance used to infer the chart configuration.

    Returns:
        Updated state including 'chart_config' or None if no data.
    """
    data_text = state.get("data") or ""
    if not data_text:
        return {**state, "chart_config": None}

    helper = trace_helper or TracingHelper()
    visualization_goal = state.get("visualization_goal") or state.get("prompt", "Chart")
    with helper.start_span(
        "chart_config_extraction",
        kind="tool",
        input_data={
            "visualization_goal": _truncate_trace_text(visualization_goal),
            "data_preview": _truncate_trace_text(data_text),
        },
    ) as span:
        formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
            data=data_text, visualization_goal=visualization_goal
        )
        response = llm.invoke(formatted_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        chart_config = _parse_chart_config(raw)
        helper.set_output(
            span,
            {
                "raw_response": _truncate_trace_text(raw),
                "chart_config": chart_config,
            },
        )
        # Do NOT include data in chart_config - it will be passed separately as DataFrame
        print("This is the chart_config: "+str(chart_config))
        return {**state, "chart_config": chart_config}


CREATE_CHART_PROMPT = """You are a Python data visualization developer creating matplotlib charts.

## TASK
Generate Python code to create a chart based on the provided configuration.

## AVAILABLE IN SCOPE
- data_df: pandas DataFrame with the data (already loaded, do NOT create it)
- config: Dictionary with chart configuration (already defined, do NOT create it)
- pd: pandas module (already imported)
- plt: matplotlib.pyplot module (already imported)

## CHART CONFIGURATION
{config}

## REQUIREMENTS
Your code must:
1. Import matplotlib.pyplot as plt
2. Import pandas as pd (if needed for data manipulation)
3. Check whether config has 'y_axes' (list), 'group_by' (string), or 'y_axis' (string) and handle accordingly:
   - If config has 'y_axes': data is in WIDE FORMAT — produce a GROUPED BAR chart, one bar series per column in y_axes (data_df[col] for each col)
   - If config has 'group_by': data is in LONG FORMAT — produce a GROUPED BAR chart by filtering data_df by each unique value of config['group_by'], using config['y_axis'] as the metric column. Use sorted unique values of data_df[config['group_by']] as series labels.
   - If config has 'y_axis' only: access data with data_df[config['y_axis']] as usual (single series)
4. Create the appropriate chart type using config['chart_type']
5. Set the chart title using config['title']
6. Add axis labels, and a legend when multiple series are present
7. Call plt.tight_layout() before plt.show()
8. Call plt.show() at the end

## CHART TYPE IMPLEMENTATIONS

### Bar Chart (chart_type='bar'):
- Use plt.bar(x_data, y_data) for vertical bars
- Good for categorical comparisons

### Line Chart (chart_type='line'):
- Use plt.plot(x_data, y_data) for lines
- Good for time series and trends

### Scatter Plot (chart_type='scatter'):
- Use plt.scatter(x_data, y_data) for points
- Good for correlations

### Area Chart (chart_type='area'):
- Use plt.fill_between(x_data, y_data) for filled areas
- Good for cumulative values

## CRITICAL: X-AXIS LABEL OVERLAP PREVENTION
**ALWAYS check and prevent x-axis label overlapping:**
- For categorical data with many categories (>10): rotate labels 45° or 90° AND use ha='right'
- For long text labels: ALWAYS rotate even if few labels
- For dates: rotate 45° with ha='right'
- If labels are still crowded after rotation: consider reducing font size with fontsize=8
- Alternative strategies:
  * Use plt.xticks(rotation=45, ha='right', fontsize=9) for crowded labels
  * Use plt.xticks(rotation=90) for very long labels
  * Consider abbreviating labels if possible
  * Increase figure width with plt.figure(figsize=(12, 6)) for many data points

## CHAIN OF THOUGHT REASONING
Before writing the code, think step by step:

**Step 1: Understanding the Configuration**
- What chart type is requested? (bar, line, scatter, area)
- Does config have 'y_axes' (list → multi-series grouped) or 'y_axis' (string → single series)?
- What is the title for the chart?
- Are there any special characteristics suggested by the column names?

**Step 2: Planning Data Extraction**
- How do I access the x-axis data? (data_df[config['x_axis']])
- Single series: data_df[config['y_axis']]
- Wide-format multi-series (y_axes): iterate over config['y_axes'], plot data_df[col] for each, using numpy offsets
- Long-format multi-series (group_by): get sorted unique values of data_df[config['group_by']], filter data_df by each value, extract config['y_axis'] values, plot using numpy offsets
- Do I need to handle special data types (dates, categories)?
- Should I sort or transform the data before plotting?

**Step 3: Selecting Matplotlib Function**
- For bar: plt.bar(x_data, y_data)
- For line: plt.plot(x_data, y_data)
- For scatter: plt.scatter(x_data, y_data)
- For area: plt.fill_between(x_data, y_data)
- What additional parameters improve readability? (marker, alpha, etc.)

**Step 4: Adding Chart Enhancements**
- Axis labels: plt.xlabel() and plt.ylabel() using config keys
- Title: plt.title() using config['title']
- **CRITICAL - Check X-axis label overlap potential:**
  * How many data points are there?
  * Are x-axis labels text (categorical) or dates?
  * Are the labels likely long (product names, location names)?
  * Decision: Apply rotation and alignment to prevent overlap
- For time series or scatter: add grid with plt.grid(True, alpha=0.3)
- Any other styling needed?

**Step 5: Finalizing and Rendering**
- Call plt.tight_layout() to prevent label cutoff (CRITICAL after rotation)
- Call plt.show() to display the chart
- Verify all requirements are met (imports, data access, chart type, labels, title)
- Double-check no syntax errors or missing steps

Now, based on this reasoning, generate the Python code.

## EXAMPLES WITH REASONING

Example 1 - Bar chart with categorical data:
config = {{"chart_type": "bar", "x_axis": "Product", "y_axis": "Sales", "title": "Sales by Product"}}

Reasoning:
- Step 1: Bar chart, x=Product (categorical), y=Sales (numeric), title provided
- Step 2: Extract data_df['Product'] and data_df['Sales'], no special handling needed
- Step 3: Use plt.bar(x_data, y_data) for vertical bars
- Step 4: Add labels, **Product names likely long → MUST rotate to prevent overlap**, apply rotation=45, ha='right'
- Step 5: tight_layout() (critical for rotated labels) then show()

Code:
import matplotlib.pyplot as plt
import pandas as pd

x_data = data_df[config['x_axis']]
y_data = data_df[config['y_axis']]

plt.figure(figsize=(10, 6))
plt.bar(x_data, y_data)
plt.xlabel(config['x_axis'])
plt.ylabel(config['y_axis'])
plt.title(config['title'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

Example 2 - Line chart with dates:
config = {{"chart_type": "line", "x_axis": "Date", "y_axis": "Revenue", "title": "Revenue Over Time"}}

Reasoning:
- Step 1: Line chart, x=Date (temporal), y=Revenue (numeric), time series visualization
- Step 2: Extract data, x-axis is dates
- Step 3: Use plt.plot(x_data, y_data) with marker='o' to show data points
- Step 4: Add labels, **dates on x-axis → rotate to prevent overlap**, add grid for trends
- Step 5: tight_layout() then show()

Code:
import matplotlib.pyplot as plt
import pandas as pd

x_data = data_df[config['x_axis']]
y_data = data_df[config['y_axis']]

plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, marker='o')
plt.xlabel(config['x_axis'])
plt.ylabel(config['y_axis'])
plt.title(config['title'])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

Example 3 - Scatter plot (no rotation needed):
config = {{"chart_type": "scatter", "x_axis": "Price", "y_axis": "Demand", "title": "Price vs Demand"}}

Reasoning:
- Step 1: Scatter plot, x=Price (numeric), y=Demand (numeric), correlation analysis
- Step 2: Extract both numeric columns, no special handling
- Step 3: Use plt.scatter(x_data, y_data) with alpha=0.6 for overlapping points
- Step 4: Add labels, **numeric x-axis → no rotation needed**, add grid for patterns
- Step 5: tight_layout() then show()

Code:
import matplotlib.pyplot as plt
import pandas as pd

x_data = data_df[config['x_axis']]
y_data = data_df[config['y_axis']]

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.6)
plt.xlabel(config['x_axis'])
plt.ylabel(config['y_axis'])
plt.title(config['title'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

Example 4 - Area chart with months:
config = {{"chart_type": "area", "x_axis": "Month", "y_axis": "Cumulative_Sales", "title": "Cumulative Sales Growth"}}

Reasoning:
- Step 1: Area chart, x=Month (temporal/sequential), y=Cumulative_Sales (numeric), shows volume
- Step 2: Extract data, ensure x is in proper order
- Step 3: Use plt.fill_between(x_data, y_data) to create filled area
- Step 4: Add labels, **month names (text) → rotate to prevent overlap**, grid for progression
- Step 5: tight_layout() then show()

Code:
import matplotlib.pyplot as plt
import pandas as pd

x_data = data_df[config['x_axis']]
y_data = data_df[config['y_axis']]

plt.figure(figsize=(12, 6))
plt.fill_between(x_data, y_data, alpha=0.4)
plt.plot(x_data, y_data)
plt.xlabel(config['x_axis'])
plt.ylabel(config['y_axis'])
plt.title(config['title'])
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

Example 5 - Bar chart with many categories:
config = {{"chart_type": "bar", "x_axis": "Store_Location", "y_axis": "Revenue", "title": "Revenue by Store Location"}}

Reasoning:
- Step 1: Bar chart, x=Store_Location (categorical, likely many stores), y=Revenue
- Step 2: Extract data, many categories expected
- Step 3: Use plt.bar(x_data, y_data)
- Step 4: **Many stores + location names (long text) → CRITICAL overlap risk**, rotate 45°, increase figure width, reduce font size
- Step 5: tight_layout() essential for rotated labels

Code:
import matplotlib.pyplot as plt
import pandas as pd

x_data = data_df[config['x_axis']]
y_data = data_df[config['y_axis']]

plt.figure(figsize=(14, 6))
plt.bar(x_data, y_data)
plt.xlabel(config['x_axis'])
plt.ylabel(config['y_axis'])
plt.title(config['title'])
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.show()


Example 6 - Grouped bar chart (multi-series, config has 'y_axes'):
config = {{"chart_type": "bar", "x_axis": "Product_Class", "y_axes": ["Avg_Revenue_Promo", "Avg_Revenue_Non_Promo"], "title": "Promo vs Non-Promo Revenue by Product Class"}}

Reasoning:
- Step 1: Bar chart, config has 'y_axes' (list of 2) → grouped bar, multi-series
- Step 2: x = data_df['Product_Class'], iterate y_axes for each series; sort by absolute difference if available
- Step 3: Use numpy arange for x positions, offset each group by bar_width
- Step 4: Add legend for series, rotate x labels, add grid
- Step 5: tight_layout() then show()

Code:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x_labels = data_df[config['x_axis']].astype(str)
y_axes = config['y_axes']
n_series = len(y_axes)
bar_width = 0.8 / n_series
x = np.arange(len(x_labels))

plt.figure(figsize=(12, 6))
for i, col in enumerate(y_axes):
    offset = (i - n_series / 2 + 0.5) * bar_width
    plt.bar(x + offset, data_df[col], width=bar_width, label=col)

plt.xlabel(config['x_axis'])
plt.ylabel('Value')
plt.title(config['title'])
plt.xticks(x, x_labels, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


Example 7 - Grouped bar from long-format data (config has 'group_by'):
config = {{"chart_type": "bar", "x_axis": "region", "y_axis": "avg_monthly_revenue", "group_by": "year", "title": "Avg Monthly Revenue by Region: 2022 vs 2023"}}

Reasoning:
- Step 1: Bar chart, config has 'group_by' = 'year' → data is in LONG FORMAT, split into series by year value
- Step 2: Get sorted unique year values (e.g., [2022, 2023]). For each year, filter data_df and extract avg_monthly_revenue indexed by region.
- Step 3: Use numpy arange for x positions, offset bars for each year group
- Step 4: Add legend with year values as labels, rotate x labels for region names
- Step 5: tight_layout() then show()

Code:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

group_col = config['group_by']
x_col = config['x_axis']
y_col = config['y_axis']

groups = sorted(data_df[group_col].unique())
x_labels = sorted(data_df[x_col].unique())
x = np.arange(len(x_labels))
n_series = len(groups)
bar_width = 0.8 / n_series

plt.figure(figsize=(12, 6))
for i, group_val in enumerate(groups):
    df_group = data_df[data_df[group_col] == group_val].set_index(x_col)
    y_vals = [df_group.loc[xl, y_col] if xl in df_group.index else 0 for xl in x_labels]
    offset = (i - n_series / 2 + 0.5) * bar_width
    plt.bar(x + offset, y_vals, width=bar_width, label=str(group_val))

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(config['title'])
plt.xticks(x, x_labels, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


## OUTPUT FORMAT
Return ONLY the Python code. No markdown formatting. No code fences. No explanations. Just the executable Python code.
"""


def create_chart(
    state: State,
    llm: ChatOllama,
    trace_helper: Optional[TracingHelper] = None,
) -> str:
    """Ask the LLM to emit matplotlib code for the given chart configuration.

    Args:
        state: Conversation state; must include 'chart_config'.
        llm: ChatOllama instance used to generate the plotting code.

    Returns:
        A Python code string (without markdown fences) that, when executed,
        renders the chart using matplotlib.
    """
    helper = trace_helper or TracingHelper()
    with helper.start_span(
        "chart_code_generation",
        kind="tool",
        input_data={"chart_config": state.get("chart_config", {})},
    ) as span:
        formatted_prompt = CREATE_CHART_PROMPT.format(config=state.get("chart_config", {}))
        response = llm.invoke(formatted_prompt)
        code = response.content if hasattr(response, "content") else str(response)
        cleaned_code = code.replace("```python", "").replace("```", "").strip()
        helper.set_output(span, {"code": _truncate_trace_text(cleaned_code)})
        # clean any accidental fences
        return cleaned_code

    
def create_visualization(state: State, llm: ChatOllama, tracer=None) -> State:
    """Create a visualization by first extracting config and then generating code.

    Uses the DataFrame directly from state (populated by lookup_sales_data).
    The generated code will reference 'data_df' directly.

    Args:
        state: Conversation state; should include 'data_df' (DataFrame).
        llm: ChatOllama instance used for config extraction and code generation.

    Returns:
        Updated state with 'chart_config', 'data_df' (DataFrame), and the generated code appended to 'answer'.
    """
    try:
        # Get DataFrame directly from state (no parsing needed!)
        data_df = state.get("data_df")

        if data_df is not None:
            print(f"Using DataFrame with shape: {data_df.shape}, columns: {list(data_df.columns)}")
        else:
            print("Warning: No DataFrame available in state")

        # Extract chart configuration
        with_config = extract_chart_config(state, llm)

        # Ensure DataFrame is in the updated state
        with_config["data_df"] = data_df

        # Generate chart code
        code = create_chart(with_config, llm)

        if tracer is not None:
            try:
                with tracer.start_as_current_span("gen_visualization", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                    span.set_input(str(state.get("prompt", "")))  # type: ignore[attr-defined]
                    span.set_output(str(code))  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
            except Exception:
                pass

        return {
            **with_config,
            "answer": with_config.get("answer", []) + [code],
        }
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


def route_to_tool(state: State) -> str:
    """Return the next node key for the graph based on 'tool_choice' in state.

    Args:
        state: Conversation state that may include 'tool_choice'.

    Returns:
        One of: 'lookup_sales_data' | 'analyzing_data' | 'create_visualization' | 'end'.
    """
    tool_choice = state.get("tool_choice", "lookup_sales_data")
    valid_tools = ["lookup_sales_data", "analyzing_data", "create_visualization", "end"]
    return tool_choice if tool_choice in valid_tools else "end"

