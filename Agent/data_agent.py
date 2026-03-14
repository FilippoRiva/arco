"""Sales Data Agent using LangGraph, DuckDB, and Ollama (LLaMA).

This module exposes a class `SalesDataAgent` that orchestrates:
- DuckDB SQL over a local parquet file
- LLM-driven tool routing (lookup → analyze → visualize)
- Chart configuration extraction and chart code generation

Usage example:
    from Agent.data_agent import SalesDataAgent

    agent = SalesDataAgent()
    result = agent.run("Show me the sales in Nov 2021")
    print(result["answer"])  # Ordered list of steps/outputs (analysis text, then code)
"""

from __future__ import annotations

import requests
import json
import os
import difflib
from functools import partial
from typing import Dict, List, Optional
import tempfile
import numpy as np
import argparse

import duckdb
import pandas as pd
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

try:
    from Agent.utils import text_to_csv, save_csv, get_evaluation_functions, make_csv_evaluator_no_gt, make_text_evaluator_no_gt, make_vis_evaluator_no_gt
    from Agent.config import AgentConfig, StepConfig
    from Agent.cache import RunCache
    from Agent.schema import DatabaseSchema, TableSchema, ColumnSchema
except ImportError:
    from utils import text_to_csv, save_csv, get_evaluation_functions, make_csv_evaluator_no_gt, make_text_evaluator_no_gt, make_vis_evaluator_no_gt
    from config import AgentConfig, StepConfig
    from cache import RunCache
    from schema import DatabaseSchema, TableSchema, ColumnSchema

# Optional energy/emissions tracking via CodeCarbon
try:
    from codecarbon import EmissionsTracker  # type: ignore
    print("CodeCarbon is available")
    _CODECARBON_AVAILABLE = True
except Exception:
    print("CodeCarbon is not available, not using it")
    EmissionsTracker = None  # type: ignore
    _CODECARBON_AVAILABLE = False

# Optional tracing/instrumentation (Phoenix / OpenInference)
try:
    from phoenix.otel import register as phoenix_register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.trace import StatusCode
    _PHOENIX_AVAILABLE = True
except Exception:  # pragma: no cover - tracing is optional
    StatusCode = None  # type: ignore
    _PHOENIX_AVAILABLE = False
    #print exception
    print(Exception)


# Mirror utils_0.py printing of langgraph version
import langgraph
import langgraph.version
print(langgraph.version)


# -----------------------------
# Constants / Defaults
# -----------------------------

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)

# -----------------------------
# State Definition
# -----------------------------

class State(TypedDict):
    prompt: str
    data: Optional[str]
    data_df: NotRequired[Optional[pd.DataFrame]]
    answer: List[str]
    visualization_goal: Optional[str]
    chart_config: Optional[dict]
    tool_choice: NotRequired[str]
    error: NotRequired[str]
    sql_query: Optional[str]
    # Per-step configuration and caching (Phase 1)
    agent_config: NotRequired[Optional[Dict]]  # AgentConfig as dict (for state flow)
    run_id: NotRequired[Optional[str]]  # Unique identifier for this execution
    cached_step_results: NotRequired[Optional[Dict]]  # Pre-loaded results from similar past runs


# -----------------------------
# LLM Helpers
# -----------------------------

_COT_SIMILARITY_THRESHOLD = 0.95

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


def select_relevant_tables(state: "State", schema: "DatabaseSchema", llm) -> List[str]:
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
    formatted_prompt = TABLE_SELECTION_PROMPT.format(
        compact_schema=schema.get_compact_summary(),
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
        return [t.name for t in schema.tables]

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

## OUTPUT FORMAT
Return ONLY the SQL query as plain text. No explanations. No markdown formatting. No code fences. Just the SQL query.
"""



def generate_sql_query(state: State, schema_context: str, llm) -> str:
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
    print("Generated SQL Query:\n", cleaned_sql)
    return cleaned_sql

# -----------------------------
# Core Step Functions (for middleware)
# -----------------------------
# These *_core functions contain just the essential logic without tracing.
# They are called by the middleware for per-step best-of-n execution.

def lookup_sales_data_core(state: State, llm, *, schema: Optional["DatabaseSchema"] = None) -> Dict:
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
        selected_names = select_relevant_tables(state, schema, llm)
        schema_context = schema.get_full_schema_str(table_names=selected_names)
    else:
        schema_context = schema.get_full_schema_str()

    # --- Generate and execute SQL ---
    sql_query = generate_sql_query(state, schema_context, llm)
    try:
        result_df = con.execute(sql_query).df()
        result_str = result_df.to_string(index=False)
        return {**state, "data": result_str, "data_df": result_df, "sql_query": sql_query}
    except Exception as e:
        print(f"Error accessing data: {str(e)}")
        return {**state, "data": "", "sql_query": sql_query, "error": f"Error accessing data: {str(e)}"}


def analyzing_data_core(state: State, llm) -> Dict:
    """Core analysis logic - LLM-based data analysis.

    Args:
        state: Conversation state; should include 'data' and 'prompt'.
        llm: LLM instance for analysis.

    Returns:
        Updated state with analysis appended to 'answer'.
    """
    try:
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(
            data=state.get("data", ""), prompt=state.get("prompt", ""), sql_query=state.get("sql_query", "")
        )
        analysis_result = llm.invoke(formatted_prompt)
        analysis_text = analysis_result.content if hasattr(analysis_result, "content") else str(analysis_result)
        return {
            **state,
            "answer": state.get("answer", []) + [analysis_text],
        }
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


def decide_tool_core(state: State, llm) -> Dict:
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
        elif len(state.get("answer", [])) > 1:
            matched_tool = "end"

        print(f"Tool selected: {matched_tool}")

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


def create_visualization_core(state: State, llm) -> Dict:
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

        # --- Validate by executing in a headless namespace (no display) ---
        # Replace plt.show() so no GUI window / inline figure is produced.
        exec_code = code.replace("plt.show()", "plt.close('all')")
        namespace: Dict = {
            "data_df": data_df,
            "config": with_config.get("chart_config", {}),
        }
        try:
            exec(exec_code, namespace)  # noqa: S102
            exec_error = ""
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
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(
            data=state.get("data", ""), prompt=state.get("prompt", ""), sql_query=state.get("sql_query","")
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
        elif len(state.get("answer", [])) > 1:
            matched_tool = "end"

        # Tracing span for tool choice (optional)
        if tracer is not None:
            try:
                with tracer.start_as_current_span("tool_choice", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                    # Minimal, robust attributes to avoid dtype issues
                    span.set_attributes({  # type: ignore[attr-defined]
                        "prompt": str(current_prompt),
                        "tool_choice": str(matched_tool),
                    })
                    span.set_input(str(current_prompt))  # type: ignore[attr-defined]
                    span.set_output(str(matched_tool))  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
            except Exception:
                pass

        print(f"Tool selected: {matched_tool}")

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
- y_axis: Column name for Y-axis (string)
- title: Descriptive chart title (string)

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


def extract_chart_config(state: State, llm: ChatOllama) -> State:
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

    visualization_goal = state.get("visualization_goal") or state.get("prompt", "Chart")
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=data_text, visualization_goal=visualization_goal
    )
    response = llm.invoke(formatted_prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    chart_config = _parse_chart_config(raw)
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
3. Access data using: data_df[config['x_axis']] and data_df[config['y_axis']]
4. Create the appropriate chart type using config['chart_type']
5. Set the chart title using config['title']
6. Add axis labels for clarity
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
- What are the x_axis and y_axis column names?
- What is the title for the chart?
- Are there any special characteristics suggested by the column names?

**Step 2: Planning Data Extraction**
- How do I access the x-axis data? (data_df[config['x_axis']])
- How do I access the y-axis data? (data_df[config['y_axis']])
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


## OUTPUT FORMAT
Return ONLY the Python code. No markdown formatting. No code fences. No explanations. Just the executable Python code.
"""


def create_chart(state: State, llm: ChatOllama) -> str:
    """Ask the LLM to emit matplotlib code for the given chart configuration.

    Args:
        state: Conversation state; must include 'chart_config'.
        llm: ChatOllama instance used to generate the plotting code.

    Returns:
        A Python code string (without markdown fences) that, when executed,
        renders the chart using matplotlib.
    """
    formatted_prompt = CREATE_CHART_PROMPT.format(config=state.get("chart_config", {}))
    response = llm.invoke(formatted_prompt)
    code = response.content if hasattr(response, "content") else str(response)
    # clean any accidental fences
    return code.replace("```python", "").replace("```", "").strip()

    
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


# -----------------------------
# Public Agent Class
# -----------------------------

class SalesDataAgent:
    """End-to-end agent to query, analyze, and visualize sales data.

    The agent builds a LangGraph with tool-selection, data lookup (DuckDB over
    parquet), LLM-based analysis, and visualization code generation. Use `run()`
    to execute a single prompt through the flow.
    """
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        streaming: bool = True,
        data_path: Optional[str] = None,
        schema: Optional["DatabaseSchema"] = None,
        ollama_url: Optional[str] = None,
        enable_tracing: bool = False,
        phoenix_api_key: Optional[str] = None,
        phoenix_endpoint: Optional[str] = None,
        project_name: str = "evaluating-agent",
        provider: str = "openai",
        openai_api_key: Optional[str] = None,
        # New: Per-step configuration and caching
        agent_config: Optional[AgentConfig] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize the agent and compile the graph.

        Args:
            model: Model name (OpenAI model like "gpt-4o-mini" or Ollama model like "llama3.2:3b").
            temperature: Sampling temperature for the LLM.
            max_tokens: Generation token limit.
            streaming: Whether to stream tokens from the LLM.
            data_path: Optional override for the parquet dataset path (single-table legacy mode).
            schema: Optional DatabaseSchema for multi-table support. When provided, takes
                    precedence over data_path for query execution. If None, auto-builds a
                    minimal schema from data_path at query time.
            ollama_url: Optional override for Ollama base URL; defaults to OLLAMA_HOST or http://localhost:11434.
            provider: LLM provider to use ("ollama" or "openai"). Default is "ollama".
            openai_api_key: Optional OpenAI API key; defaults to OPENAI_API_KEY env var.
            agent_config: Optional AgentConfig for per-step hyperparameter control.
            cache_dir: Optional directory for caching run results.
        """
        self.provider = provider.lower()

        if self.provider == "openai":
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided via openai_api_key parameter or OPENAI_API_KEY environment variable")
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                api_key=api_key,
            )
            self.ollama_url = None
        else:  # ollama
            self.ollama_url = ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                base_url=self.ollama_url,
            )

        self.data_path = data_path or DEFAULT_DATA_PATH
        # Multi-table schema. None means single-table legacy mode (auto-built at query time).
        self.schema = schema

        # Optional Phoenix/OpenInference tracing integration
        self.tracer = None
        self.tracing_enabled = False
        if enable_tracing and _PHOENIX_AVAILABLE:
            try:
                # Environment variables similar to utils_0.py
                if phoenix_api_key:
                    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
                    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"
                if phoenix_endpoint:
                    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

                tracer_provider = phoenix_register(
                    project_name=project_name,
                    endpoint=(phoenix_endpoint or "https://app.phoenix.arize.com/v1/traces"),
                )
                LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)
                self.tracer = tracer_provider.get_tracer(__name__)
                self.tracing_enabled = True
            except Exception as _:
                self.tracer = None
                self.tracing_enabled = False

        # Store model parameters for LLM factory method
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Initialize per-step configuration
        if agent_config is not None:
            self.agent_config = agent_config
        else:
            # Create default config with current parameters
            self.agent_config = AgentConfig(
                model=model,
                provider=provider,
                ollama_url=self.ollama_url or "http://localhost:11434",
                openai_api_key=self.openai_api_key,
            )

        # Initialize result cache
        self.cache = RunCache(cache_dir or "./cache/agent_runs")

        # Track step results during execution (for caching)
        self.current_run_step_results: Dict[str, List[Dict]] = {}

        self.graph = self._build_graph()
        self.run_checked = False

    def check_ollama(self):
        try:
            self.llm.invoke("Hello, how are you?")
            print("Ollama is running locally")
            return True
        except Exception as e:
            print(e)
            return False

    def check_model(self):
        """Check if the model is running locally (Ollama) or accessible (OpenAI)"""
        if self.provider == "openai":
            try:
                self.llm.invoke("Hello")
                print("OpenAI API is accessible")
                return True
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return False
        else:
            try:
                base = self.ollama_url.rstrip("/")
                requests.get(f"{base}/api/version", timeout=3).json()
                print("Server is running locally")
                return self.check_ollama()
            except Exception as e:
                print(e)
                return False

    def _create_llm(
        self,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        num_beams: int = 1,
        no_repeat_ngram_size: Optional[int] = None,
    ):
        """Factory method to create LLM instances with specific parameters.

        Creates a new LLM instance instead of mutating the global self.llm,
        which allows per-step parameter customization.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter (skipped for OpenAI)
            num_beams: Beam search width, 1 = greedy/disabled (skipped for OpenAI)
            no_repeat_ngram_size: Prevent repeating n-grams of this size (skipped for OpenAI)

        Returns:
            ChatOllama or ChatOpenAI instance configured with the given parameters
        """
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=self.streaming,
                api_key=self.openai_api_key,
                top_p=top_p,
            )
        else:
            kwargs = dict(
                model=self.model,
                temperature=temperature,
                num_predict=max_tokens,
                streaming=self.streaming,
                base_url=self.ollama_url,
                top_p=top_p,
            )
            if top_k is not None:
                kwargs["top_k"] = top_k
            if num_beams > 1:
                kwargs["num_beams"] = num_beams
            if no_repeat_ngram_size is not None:
                kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            return ChatOllama(**kwargs)

    def _apply_cot_iterations(
        self,
        step_name: str,
        state: State,
        core_fn,
        llm,
        initial_result: Dict,
        cot_n: int,
    ) -> Dict:
        """Apply up to cot_n iterative CoT refinement steps to a single LLM call.

        Starting from initial_result, repeatedly re-invokes core_fn with a
        CoTRefinementLLM that appends the previous response to every prompt.
        Stops early when the output converges (similarity >= _COT_SIMILARITY_THRESHOLD)
        or after cot_n total iterations (including the initial one).

        Args:
            step_name: Name of the step (for logging and output extraction).
            state: Current agent state (unchanged across iterations).
            core_fn: The core step function, signature (state, llm) -> Dict.
            llm: The base LLM instance (temperature already set by the caller).
            initial_result: Result from the first (non-refinement) call.
            cot_n: Maximum total number of iterations (1 = no refinement).

        Returns:
            The result from the final (or converged) iteration.
        """
        if cot_n <= 1:
            return initial_result

        result = initial_result
        previous_output = _extract_step_output(step_name, result)
        execution_error = result.get("error", "")

        for cot_i in range(1, cot_n):
            refinement_llm = CoTRefinementLLM(llm, previous_output, execution_error)
            try:
                new_result = core_fn(state, refinement_llm)
            except Exception as e:
                print(f"[{step_name}] CoT iteration {cot_i + 1}/{cot_n} failed: {e}")
                break

            new_error = new_result.get("error", "")
            if new_error:
                print(
                    f"[{step_name}] CoT iteration {cot_i + 1}/{cot_n}: "
                    f"execution error — {new_error}"
                )
                result = new_result
                previous_output = _extract_step_output(step_name, new_result)
                execution_error = new_error
                continue

            new_output = _extract_step_output(step_name, new_result)
            ratio = difflib.SequenceMatcher(None, previous_output, new_output).ratio()
            print(
                f"[{step_name}] CoT iteration {cot_i + 1}/{cot_n}: "
                f"similarity={ratio:.3f}"
            )

            result = new_result
            execution_error = ""

            if ratio >= _COT_SIMILARITY_THRESHOLD:
                print(
                    f"[{step_name}] CoT early stop: output converged "
                    f"(similarity={ratio:.3f} >= {_COT_SIMILARITY_THRESHOLD})"
                )
                break

            previous_output = new_output

        return result

    @staticmethod
    def _run_gt_eval(
        step_name: str,
        config: "StepConfig",
        result: Dict,
        state: Dict,
        all_results: Optional[List[Dict]] = None,
    ) -> None:
        """Run ground-truth evaluation for tracking/logging only.

        This NEVER influences selection — it only logs GT scores on the
        already-selected result so performance can be tracked without
        steering the agent.
        """
        if config.gt_eval_fn is None:
            return

        # Score the selected (best) result
        try:
            gt_score = config.gt_eval_fn(result, state)
            result["_gt_score"] = gt_score
            print(f"[{step_name}] GT tracking score: {gt_score:.3f}")
        except Exception as e:
            print(f"[{step_name}] GT eval error (tracking only): {e}")

        # Score all N candidates for richer tracking
        if all_results and len(all_results) > 1:
            all_gt_scores = []
            for r in all_results:
                try:
                    all_gt_scores.append(config.gt_eval_fn(r, state))
                except Exception:
                    all_gt_scores.append(0.0)
            result["_all_gt_scores"] = all_gt_scores
            print(f"[{step_name}] All GT scores: {[f'{s:.3f}' for s in all_gt_scores]}")

    def _execute_step_with_config(
        self,
        step_name: str,
        state: State,
        core_fn,
        config: StepConfig,
    ) -> Dict:
        """Execute a step with per-step best-of-n, evaluation, and caching.

        This middleware method:
        1. Checks cache if config.use_cache is True
        2. Runs best-of-n sampling if cache miss or force_fresh
        3. Evaluates each of N runs using config.eval_fn
        4. Selects best result using config.selection_fn
        5. Stores all N results for caching

        Args:
            step_name: Name of the step (for logging and caching)
            state: Current agent state
            core_fn: The core step function, signature: (state, llm) -> Dict
            config: StepConfig with parameters for this step

        Returns:
            Updated state dict from the best run
        """
        # Check if step is enabled
        if not config.enabled:
            print(f"[{step_name}] Step disabled, skipping")
            return dict(state)

        # Check cache first
        if config.use_cache and config.cache_mode != "force_fresh":
            cached_results = state.get("cached_step_results", {})
            if cached_results and step_name in cached_results:
                cached = cached_results[step_name]
                print(f"[{step_name}] Found {len(cached)} cached result(s)")

                # Preserve cached_step_results from the current state so that
                # subsequent steps can still find their own cached results.
                # (The cached dict was produced in a previous run whose state
                # had an empty or different cached_step_results.)
                live_csr = state.get("cached_step_results", {})

                if config.cache_mode == "skip":
                    # Use first cached result directly (previously selected best)
                    print(f"[{step_name}] Using cached result (skip mode)")
                    if cached:
                        result = dict(cached[0])
                        result["cached_step_results"] = live_csr
                        self._run_gt_eval(step_name, config, result, state)
                        return result
                    return dict(state)

                # cache_mode == "auto": Re-evaluate cached results with current eval_fn
                if config.eval_fn and len(cached) > 1:
                    scores = []
                    for r in cached:
                        try:
                            score = config.eval_fn(r, state)
                        except Exception:
                            score = 0.0
                        scores.append(score)
                    best_idx = config.selection_fn(scores)
                    print(f"[{step_name}] Re-selected cached result {best_idx + 1}/{len(cached)}")
                    result = dict(cached[best_idx])
                    result["cached_step_results"] = live_csr
                    self._run_gt_eval(step_name, config, result, state, all_results=cached)
                    return result
                elif cached:
                    result = dict(cached[0])
                    result["cached_step_results"] = live_csr
                    self._run_gt_eval(step_name, config, result, state)
                    return result

        # No cache or force_fresh: run the step
        n = config.n
        temps = config.get_temperatures()

        if n == 1:
            # Simple case: single run
            llm = self._create_llm(
                temperature=temps[0],
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
            )
            try:
                result = core_fn(state, llm)
                result["_temperature"] = temps[0]
                result["_run_idx"] = 0
                result = self._apply_cot_iterations(step_name, state, core_fn, llm, result, config.cot_n)
            except Exception as e:
                print(f"[{step_name}] Error: {e}")
                result = dict(state)
                result["error"] = str(e)

            self.current_run_step_results[step_name] = [result]
            self._run_gt_eval(step_name, config, result, state)
            return result

        # Best-of-n execution
        results = []
        scores = []

        print(f"[{step_name}] Running best-of-{n} with temps {[f'{t:.2f}' for t in temps]}")

        for i, temp in enumerate(temps):
            llm = self._create_llm(
                temperature=temp,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
            )

            try:
                result = core_fn(state, llm)
                result["_temperature"] = temp
                result["_run_idx"] = i
                result = self._apply_cot_iterations(step_name, state, core_fn, llm, result, config.cot_n)

                # Evaluate if function provided
                if config.eval_fn:
                    try:
                        score = config.eval_fn(result, state)
                    except Exception as eval_err:
                        print(f"  Run {i + 1}/{n}: eval error: {eval_err}")
                        score = 0.0
                else:
                    score = 0.0

                results.append(result)
                scores.append(score)
                print(f"  Run {i + 1}/{n} (T={temp:.2f}): score={score:.3f}")

            except Exception as e:
                print(f"  Run {i + 1}/{n} failed: {e}")
                error_result = dict(state)
                error_result["error"] = str(e)
                error_result["_temperature"] = temp
                error_result["_run_idx"] = i
                results.append(error_result)
                scores.append(-float('inf'))

        # Store all results for caching
        self.current_run_step_results[step_name] = results

        # Batch re-evaluation if batch_eval_fn is provided (e.g. consensus scoring)
        if config.batch_eval_fn:
            try:
                scores = config.batch_eval_fn(results, state)
                print(f"[{step_name}] Batch eval scores: {[f'{s:.3f}' for s in scores]}")
            except Exception as e:
                print(f"[{step_name}] Batch eval error: {e}")

        # Select best result
        if not scores or all(s == -float('inf') for s in scores):
            best_result = results[0] if results else dict(state)
        else:
            best_idx = config.selection_fn(scores)
            best_result = results[best_idx]
            best_result["_best_idx"] = best_idx
            best_result["_all_scores"] = scores
            print(f"[{step_name}] Selected run {best_idx + 1}/{n} (score={scores[best_idx]:.3f})")

        self._run_gt_eval(step_name, config, best_result, state, all_results=results)
        return best_result

    def _maybe_save_run_results(
        self,
        run_id: str,
        prompt: str,
        result: Dict,
        save_results: bool
    ) -> None:
        """Save run results to cache if save_results is True.

        Args:
            run_id: Unique identifier for this run
            prompt: User prompt that initiated this run
            result: Final result from the agent
            save_results: Whether to actually save
        """
        if not save_results:
            return

        try:
            self.cache.save_run(
                run_id=run_id,
                prompt=prompt,
                agent_config=self.agent_config.to_dict(),
                step_results=self.current_run_step_results,
                final_result=result,
                metadata={}
            )
            print(f"[Agent] Run saved with ID: {run_id}")
        except Exception as e:
            print(f"[Agent] Warning: Failed to save run to cache: {e}")

    def _build_graph(self):
        """Construct and compile the LangGraph for the agent run loop.

        Uses the middleware pattern to support per-step configuration including
        best-of-n sampling, custom evaluation, and caching. Each node wraps
        a *_core function with _execute_step_with_config().
        """
        graph = StateGraph(State)

        # Factory to create configured node functions
        def make_configured_node(step_name: str, core_fn):
            """Create a node function that uses per-step configuration."""
            def node_fn(state: State) -> Dict:
                config = self.agent_config.get_step_config(step_name)
                return self._execute_step_with_config(step_name, state, core_fn, config)
            return node_fn

        # Bind schema into lookup_sales_data_core via partial so the middleware
        # signature core_fn(state, llm) is preserved.
        lookup_core_with_schema = partial(lookup_sales_data_core, schema=self.schema)

        # Add nodes with configuration wrappers
        graph.add_node("decide_tool", make_configured_node("decide_tool", decide_tool_core))
        graph.add_node("lookup_sales_data", make_configured_node("lookup_sales_data", lookup_core_with_schema))
        graph.add_node("analyzing_data", make_configured_node("analyzing_data", analyzing_data_core))
        graph.add_node("create_visualization", make_configured_node("create_visualization", create_visualization_core))

        graph.set_entry_point("decide_tool")

        # Routing logic (unchanged)
        graph.add_conditional_edges(
            "decide_tool",
            route_to_tool,
            {
                "lookup_sales_data": "lookup_sales_data",
                "analyzing_data": "analyzing_data",
                "create_visualization": "create_visualization",
                "end": END,
            },
        )

        graph.add_edge("lookup_sales_data", "decide_tool")
        graph.add_edge("analyzing_data", "decide_tool")
        graph.add_edge("create_visualization", "decide_tool")

        return graph.compile()
    
    def draw_graph(self) -> str:
        """Return an ASCII rendering of the compiled graph if available."""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # Fallback if mermaid is not available
            print(self.graph.get_graph().print_ascii())

    def run_core(
        self,
        prompt: str,
        *,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        # New: caching parameters
        run_id: Optional[str] = None,
        cached_step_results: Optional[Dict] = None,
        save_results: bool = False,
    ) -> Dict:
        """Execute the agent for a single prompt.

        Args:
            prompt: Natural-language request or question.
            visualization_goal: Optional explicit goal for charts; defaults to the prompt.
            lookup_only: Only run data lookup step.
            no_vis: Skip visualization step.
            run_id: Unique ID for this run (for caching).
            cached_step_results: Pre-loaded cached results from similar past runs.
            save_results: Whether to save this run's results to cache.

        Returns:
            The final state dictionary produced by the compiled graph execution.
        """
        import uuid

        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        # Reset step results tracker
        self.current_run_step_results = {}

        # Initialize state with caching info
        state = {
            "prompt": prompt,
            "run_id": run_id,
            "cached_step_results": cached_step_results or {},
        }
        if not self.run_checked:
            print("Checking the model can run locally")
            self.run_checked = self.check_model()
        
        if not self.run_checked:
            error_msg = "Model is not accessible. " + (
                "Remember to run 'ollama serve' for Ollama models." if self.provider == "ollama"
                else "Check your OpenAI API key and internet connection."
            )
            print(error_msg)
            return {**state, "error": error_msg}
    
        if lookup_only:
            print("[Agent] Running only lookup_sales_data")
            try:
                if self.tracing_enabled and self.tracer is not None:
                    with self.tracer.start_as_current_span("AgentRun_LookupOnly", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                        span.set_input(state)  # type: ignore[attr-defined]
                        result = lookup_sales_data(state, self.llm, self.tracer, schema=self.schema)
                        span.set_output(result)  # type: ignore[attr-defined]
                        if StatusCode is not None:
                            span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                        self.current_run_step_results["lookup_sales_data"] = [dict(result)]
                        self._maybe_save_run_results(run_id, prompt, result, save_results)
                        result["run_id"] = run_id
                        return result
                else:
                    result = lookup_sales_data(state, self.llm, schema=self.schema)
                    self.current_run_step_results["lookup_sales_data"] = [dict(result)]
                    self._maybe_save_run_results(run_id, prompt, result, save_results)
                    result["run_id"] = run_id
                    return result
            except Exception as _e:
                return {**state, "error": f"Lookup failed: {str(_e)}"}
        if no_vis:
            print("[Agent] Running agent without visualization")
            try:
                lookup_cfg = self.agent_config.get_step_config("lookup_sales_data")
                analyzing_cfg = self.agent_config.get_step_config("analyzing_data")
                lookup_core = partial(lookup_sales_data_core, schema=self.schema)
                if self.tracing_enabled and self.tracer is not None:
                    with self.tracer.start_as_current_span("AgentRun_NoVis", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                        span.set_input(state)  # type: ignore[attr-defined]
                        state = self._execute_step_with_config("lookup_sales_data", state, lookup_core, lookup_cfg)
                        result = self._execute_step_with_config("analyzing_data", state, analyzing_data_core, analyzing_cfg)
                        print(f"\nAgent response: {result.get('answer', [None])[0]}")
                        span.set_output(result)  # type: ignore[attr-defined]
                        if StatusCode is not None:
                            span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                        self._maybe_save_run_results(run_id, prompt, result, save_results)
                        result["run_id"] = run_id
                        return result
                else:
                    state = self._execute_step_with_config("lookup_sales_data", state, lookup_core, lookup_cfg)
                    result = self._execute_step_with_config("analyzing_data", state, analyzing_data_core, analyzing_cfg)
                    print(f"\nAgent response: {result.get('answer', [None])[0]}")
                    self._maybe_save_run_results(run_id, prompt, result, save_results)
                    result["run_id"] = run_id
                    return result
            except Exception as _e:
                print(f"Lookup failed: {str(_e)}")
                return {**state, "error": f"Lookup failed: {str(_e)}"}
        
        if visualization_goal:
            state["visualization_goal"] = visualization_goal
        print("Running the graph...")
        if self.tracing_enabled and self.tracer is not None:
            try:
                with self.tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                    print("[LangGraph] Starting LangGraph execution with tracing")
                    span.set_input(state)  # type: ignore[attr-defined]
                    result = self.graph.invoke(state)
                    print(f"\nAgent response: {result.get('answer', [])}")
                    span.set_output(result)  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                    print("[LangGraph] LangGraph execution completed")
                    self._maybe_save_run_results(run_id, prompt, result, save_results)
                    result["run_id"] = run_id
                    return result
            except Exception:
                # Fallback to non-traced execution on any tracing error
                result = self.graph.invoke(state)
                print(f"\nAgent response: {result.get('answer', [])}")
                self._maybe_save_run_results(run_id, prompt, result, save_results)
                result["run_id"] = run_id
                return result
        else:
            print("[LangGraph] Starting LangGraph execution")
            result = self.graph.invoke(state)
            print("[LangGraph] LangGraph execution completed")
            self._maybe_save_run_results(run_id, prompt, result, save_results)
            result["run_id"] = run_id
            return result
    
    def _run_with_evaluation(
        self,
        *,
        prompt: str,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        best_of_n: int = 1,
        temp: Optional[float] = None,
        temp_max: Optional[float] = None,
        csv_eval_fn: Optional[callable] = None,
        text_eval_fn: Optional[callable] = None,
        vis_eval_fn: Optional[callable] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """Core evaluation logic extracted from run() for CodeCarbon wrapping."""
        
        if best_of_n > 1 and temp is not None and temp_max is not None:
            temps = np.linspace(temp, temp_max, best_of_n).tolist()
        else:
            temps = [temp if temp is not None else self.llm.temperature] * best_of_n
        
        print(f"[Agent] Running best-of-{best_of_n} with temperatures: {temps}")
        
        all_results = []
        all_scores = []
        
        for i in range(best_of_n):
            original_temp = self.llm.temperature
            self.llm.temperature = temps[i]
            
            try:
                result = self.run_core(
                    prompt,
                    visualization_goal=visualization_goal,
                    lookup_only=lookup_only,
                    no_vis=no_vis
                )

                # Save CSV
                csv_path = None
                if result.get("data"):
                    csv_path = os.path.join(save_dir, f"run_data.csv")
                    result_rows = text_to_csv(result['data'])
                    save_csv(result_rows, csv_path)
                
                # Extract analysis text
                analysis_text = result.get("answer", [None])[0] if result.get("answer") else None
                
                # Evaluate
                score = 0.0
                csv_score = None
                text_score = None
                
                if csv_eval_fn:
                    csv_score = csv_eval_fn(csv_path)
                    score += csv_score
                    result["csv_score"] = csv_score
                
                if text_eval_fn:
                    text_score = text_eval_fn(analysis_text)
                    score += text_score
                    result["text_score"] = text_score

                # Visualization evaluation
                if vis_eval_fn and not no_vis and not lookup_only:
                    chart_config = result.get("chart_config")
                    # Chart code is the last answer entry (after analysis text)
                    answers = result.get("answer", [])
                    chart_code = answers[-1] if len(answers) > 1 else None

                    if chart_config and chart_code:
                        vis_score = vis_eval_fn(chart_config, chart_code)
                        score += vis_score
                        result["vis_score"] = vis_score

                result["temperature"]= temps[i]

                all_results.append(result)
                all_scores.append(score)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
        self.llm.temperature = original_temp
        print(all_scores)
        if not all_scores:
            return {}, 0.0
        
        best_idx = int(np.argmax(all_scores))
        best_result = all_results[best_idx]
        
        results_path = os.path.join(save_dir, "all_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        score_variance = (max(all_scores) - min(all_scores))/max(all_scores) if max(all_scores) != 0 else 0.0
        return best_result, score_variance
            
    def run(
        self,
        prompt: str,
        *,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        best_of_n: int = 1,
        temp: Optional[float] = None,
        temp_max: Optional[float] = None,
        csv_eval_fn: Optional[callable] = None,
        text_eval_fn: Optional[callable] = None,
        vis_eval_fn: Optional[callable] = None,
        save_dir: Optional[str] = None,
        enable_codecarbon: bool = False,
        # New: caching parameters
        run_id: Optional[str] = None,
        reuse_from: Optional[str] = None,
        step_overrides: Optional[Dict[str, Dict]] = None,
        save_results: bool = False,
    ) -> Dict:
        """Run the agent with optional caching and per-step configuration.

        Args:
            prompt: User query/question.
            visualization_goal: Optional explicit visualization goal.
            lookup_only: Only run data lookup step.
            no_vis: Skip visualization step.
            best_of_n: (Deprecated) Number of agent-level runs for old best-of-n.
                       Use step-level configuration via AgentConfig instead.
            temp, temp_max: (Deprecated) Temperature range for old best-of-n.
            csv_eval_fn, text_eval_fn, vis_eval_fn: (Deprecated) Evaluation functions
                       for old agent-level best-of-n.
            save_dir: Directory for saving results (old API).
            enable_codecarbon: Enable carbon emissions tracking.
            reuse_from: Run ID to load cached results from (new caching API).
            step_overrides: Dict mapping step_name -> config overrides for this run.
                           Example: {"analyzing_data": {"n": 10, "temp_max": 0.9}}
            save_results: Whether to save this run's results to cache.

        Returns:
            Result dict with 'answer', 'data', 'chart_config', etc.
            Includes 'run_id' if save_results=True.
        """

        if save_dir is None:
            save_dir = tempfile.mkdtemp(prefix="agent_runs_")
        os.makedirs(save_dir, exist_ok=True)

        # Apply step overrides if provided
        original_config = None
        if step_overrides:
            from copy import deepcopy
            original_config = self.agent_config
            self.agent_config = deepcopy(self.agent_config)
            for step_name, overrides in step_overrides.items():
                step_config = self.agent_config.get_step_config(step_name)
                for key, value in overrides.items():
                    if hasattr(step_config, key):
                        setattr(step_config, key, value)

        # Find/load cached results
        cached_step_results = {}
        if reuse_from:
            # Load from specific run
            cached_step_results = self.cache.load_all_step_results(reuse_from)
            if cached_step_results:
                print(f"[Agent] Loaded cached results from run: {reuse_from}")
            else:
                print(f"[Agent] Warning: No cached results found for run: {reuse_from}")
        elif save_results:
            # Auto-find similar runs
            similar_runs = self.cache.find_similar_runs(prompt, top_k=3)
            if similar_runs:
                print(f"[Agent] Found {len(similar_runs)} similar run(s): {similar_runs}")
                cached_step_results = self.cache.load_all_step_results(similar_runs[0])

        # Use new API if using caching features and not using old best-of-n
        if (save_results or reuse_from) and best_of_n == 1:
            try:
                result = self.run_core(
                    prompt,
                    visualization_goal=visualization_goal,
                    lookup_only=lookup_only,
                    no_vis=no_vis,
                    run_id=run_id,
                    cached_step_results=cached_step_results,
                    save_results=save_results,
                )
                return result
            finally:
                # Restore original config if we modified it
                if original_config is not None:
                    self.agent_config = original_config

        # Restore original config before falling through to old API
        if original_config is not None:
            self.agent_config = original_config

        # Wrap execution with CodeCarbon if requested and available
        if enable_codecarbon and _CODECARBON_AVAILABLE:
            codecarbon_dir = os.path.join(save_dir, "codecarbon")
            os.makedirs(codecarbon_dir, exist_ok=True)
            try:
                with EmissionsTracker(  # type: ignore[call-arg]
                    project_name="SalesDataAgent",
                    output_dir=codecarbon_dir,
                    save_to_file=True,
                    measure_power_secs=1,
                    log_level="error",
                ):
                    return self._run_with_evaluation(
                        prompt=prompt,
                        visualization_goal=visualization_goal,
                        lookup_only=lookup_only,
                        no_vis=no_vis,
                        best_of_n=best_of_n,
                        temp=temp,
                        temp_max=temp_max,
                        csv_eval_fn=csv_eval_fn,
                        text_eval_fn=text_eval_fn,
                        vis_eval_fn=vis_eval_fn,
                        save_dir=save_dir,
                    )
            except Exception as e:
                print(f"CodeCarbon tracking failed: {e}, continuing without it")
                # Fall through to run without CodeCarbon

        return self._run_with_evaluation(
            prompt=prompt,
            visualization_goal=visualization_goal,
            lookup_only=lookup_only,
            no_vis=no_vis,
            best_of_n=best_of_n,
            temp=temp,
            temp_max=temp_max,
            csv_eval_fn=csv_eval_fn,
            text_eval_fn=text_eval_fn,
            vis_eval_fn=vis_eval_fn,
            save_dir=save_dir,
        )

__all__ = ["SalesDataAgent", "State"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the Sales Data Agent")
    parser.add_argument("prompt", type=str, help="User prompt/question")
    parser.add_argument("--gt_csv", type=str, default=None, help="Path to ground-truth CSV file")
    parser.add_argument("--gt_text", type=str, default=None, help="Path to a text file containing the ground-truth")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save run results")

    parser.add_argument("--data", dest="data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to parquet file")
    parser.add_argument("--goal", dest="visualization_goal", type=str, default=None, help="Optional visualization goal")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name (default: gpt-4o-mini)")
       
    # Agent type options
    agent_group = parser.add_mutually_exclusive_group()
    agent_group.add_argument("--lookup_only", action="store_true", help="Only run data lookup")
    agent_group.add_argument("--no_vis", action="store_true", help="Run lookup then analysis (no visualization)")

    # Best-of-n options
    parser.add_argument("--best_of_n", type=int, default=1, help="Run agent N times and pick the best result")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature used to build the agent and as minimum for best-of-n")
    parser.add_argument("--temp-max", type=float, default=None, help="Max temperature for best-of-n, if not provided best-of-n runs without modifying the temperature")
    parser.add_argument("--cot_n", type=int, default=1, help="Number of iterative CoT refinement steps per LLM call (1 = no refinement, N = up to N iterations with early stop on convergence)")

    # CSV evaluation options
    csv_eval_group = parser.add_mutually_exclusive_group()
    csv_eval_group.add_argument("--py_csv_eval", action="store_true", help="Use Python evaluator for CSV IoU")
    csv_eval_group.add_argument("--cpp_csv_eval", action="store_true", help="Use C++ evaluator for CSV IoU")
    parser.add_argument("--evaluator_exe", type=str, default=None, help="Path to C++ comparator executable")
    parser.add_argument("--eval_keys", type=str, default=None, help="Comma-separated key columns for C++ comparator")
    parser.add_argument("--iou_type", type=str, default="rows", choices=["columns", "rows", "table"], help="Type of IoU to use for CSV evaluation, choose between 'columns', 'rows', 'table'")

    # Text evaluation options
    text_eval_group = parser.add_mutually_exclusive_group()
    text_eval_group.add_argument("--spice_text_eval", action="store_true")
    text_eval_group.add_argument("--bleu_text_eval", action="store_true")
    text_eval_group.add_argument("--llm_text_eval", action="store_true")
    parser.add_argument("--bleu_nltk", action="store_true", help="Use nltk for BLEU implementation instead of simple BLEU")
    parser.add_argument("--spice_jar", type=str, default=None, help="Path to SPICE jar (e.g., spice-1.0.jar)")
    parser.add_argument("--spice_java_bin", type=str, default="java", help="Java executable for SPICE")

    # Visualization evaluation options
    parser.add_argument("--vis_eval", action="store_true", help="Enable visualization evaluation using LLM-as-a-judge")
    parser.add_argument("--gt_vis_path", type=str, default=None, help="Path to visualization ground truth JSON file")
    parser.add_argument("--vis_judge_model", type=str, default="gpt-5.1", help="Model for visualization judge (default: gpt-5.1)")
    parser.add_argument("--vis_provider", type=str, default="openai", choices=["openai", "ollama"], help="Provider for visualization judge")

    # Phoenix tracking options
    parser.add_argument("--enable_tracing", action="store_true", help="Enable Phoenix tracing/tracking")
    parser.add_argument("--phoenix_endpoint", type=str, default="http://localhost:6006/v1/traces", help="Phoenix endpoint URL (default: https://app.phoenix.arize.com/v1/traces)")
    parser.add_argument("--project_name", type=str, default="evaluating-agent", help="Phoenix project name")

    # CodeCarbon options
    parser.add_argument("--enable_codecarbon", action="store_true", help="Enable CodeCarbon energy/emissions tracking")
    
    args = parser.parse_args()

    # Create agent
    agent = SalesDataAgent(
        model=args.model,
        temperature=args.temp,
        data_path=args.data_path,
        enable_tracing=args.enable_tracing,
        phoenix_endpoint=args.phoenix_endpoint,
        project_name=args.project_name,
    )

    # Apply cot_n to all step configs when requested
    if args.cot_n > 1:
        for _step in ['decide_tool', 'lookup_sales_data', 'analyzing_data', 'create_visualization']:
            _cfg = agent.agent_config.get_step_config(_step)
            _cfg.cot_n = args.cot_n
            agent.agent_config.set_step_config(_step, _cfg)
        print(f"[Agent] CoT iterative refinement enabled: cot_n={args.cot_n}")

    # Load visualization ground truth if provided
    gt_vis_config = None
    gt_vis_code = None
    vis_goal = None
    explicit_requirements = None
    if args.gt_vis_path:
        try:
            with open(args.gt_vis_path, 'r', encoding='utf-8') as f:
                vis_gt_data = json.load(f)
                # If it's a list, use the first entry (for single-query evaluation)
                if isinstance(vis_gt_data, list) and len(vis_gt_data) > 0:
                    vis_gt_entry = vis_gt_data[0]
                else:
                    vis_gt_entry = vis_gt_data
                gt_vis_config = vis_gt_entry.get("gt_chart_config")
                gt_vis_code = vis_gt_entry.get("gt_chart_code")
                vis_goal = vis_gt_entry.get("visualization_goal")
                explicit_requirements = vis_gt_entry.get("explicit_requirements")
        except Exception as e:
            print(f"Failed to load visualization ground truth: {e}")

    # Get evaluation functions based on arguments
    csv_eval_fn, text_eval_fn, vis_eval_fn = get_evaluation_functions(
        lookup_only=args.lookup_only,
        gt_csv_path=args.gt_csv,
        py_csv_eval=args.py_csv_eval,
        cpp_csv_eval=args.cpp_csv_eval,
        evaluator_exe=args.evaluator_exe,
        eval_keys=args.eval_keys,
        gt_text_path=args.gt_text,
        iou_type=args.iou_type,
        spice_text_eval=args.spice_text_eval,
        bleu_text_eval=args.bleu_text_eval,
        llm_text_eval=args.llm_text_eval,
        bleu_nltk=args.bleu_nltk,
        spice_jar=args.spice_jar,
        spice_java_bin=args.spice_java_bin,
        # Visualization evaluation options
        vis_eval=args.vis_eval,
        gt_vis_config=gt_vis_config,
        gt_vis_code=gt_vis_code,
        vis_goal=vis_goal or args.visualization_goal,
        explicit_requirements=explicit_requirements,
        vis_judge_model=args.vis_judge_model,
        vis_provider=args.vis_provider,
    )

    # Run agent
    output, score_variance = agent.run(
        args.prompt,
        visualization_goal=args.visualization_goal,
        lookup_only=args.lookup_only,
        no_vis=args.no_vis,
        best_of_n=args.best_of_n,
        temp=args.temp,
        temp_max=args.temp_max,
        csv_eval_fn=csv_eval_fn,
        text_eval_fn=text_eval_fn,
        vis_eval_fn=vis_eval_fn,
        save_dir=args.save_dir,
        enable_codecarbon=args.enable_codecarbon,
    )
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if args.best_of_n > 1:
        print(f"Score variance: {score_variance:.4f}")
    print(f"Answer: {output.get('answer', [])}")
    if args.save_dir or args.best_of_n > 1:
        print(f"Results saved to: {args.save_dir or 'temp directory'}")