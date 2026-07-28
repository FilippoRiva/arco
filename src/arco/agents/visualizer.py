from typing import TYPE_CHECKING

from arco.core import Agent, AgentException, AgentType
from arco.evaluators import VisualizerEvaluator

if TYPE_CHECKING:
    from arco.core import LLM, Evaluator, State

import logging

logger = logging.getLogger(__name__)


class Visualizer(Agent):
    _CHART_CONFIGURATION_PROMPT = """You are a data visualization expert designing chart configurations.

## TASK
Create a JSON configuration object for visualizing the provided data according to the data analysis and visualization request from the user prompt.

## USER PROMPT
{prompt}

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

## EXAMPLES

Example 1 - Single time series (simple line):
    Data columns: Date, Revenue
    Goal: "Show revenue trends over time"
    Output: {{"chart_type": "line", "x_axis": "Date", "y_axis": "Revenue", "title": "Revenue Trends Over Time"}}

Example 2 - Wide-format multi-series (grouped bar with y_axes):
    Data columns: Product_Class, Avg_Revenue_Promo, Avg_Revenue_Non_Promo
    Goal: "Compare average revenue per unit during promotions vs non-promotions for each product class"
    Output: {{"chart_type": "bar", "x_axis": "Product_Class", "y_axes": ["Avg_Revenue_Promo", "Avg_Revenue_Non_Promo"], "title": "Promo vs Non-Promo Revenue by Product Class"}}

Example 3 - Long-format multi-series (grouped bar with group_by):
    Data columns: region, year, avg_monthly_revenue (8 rows: 4 regions x 2 years, long format)
    Goal: "Compare average monthly revenue by region for 2022 vs 2023"
    Output: {{"chart_type": "bar", "x_axis": "region", "y_axis": "avg_monthly_revenue", "group_by": "year", "title": "Avg Monthly Revenue by Region: 2022 vs 2023"}}

## OUTPUT FORMAT
Return ONLY a valid JSON object. No markdown. No code fences. No backticks. No explanations. Just the JSON.
"""

    _CREATE_CHART_PROMPT = """You are a Python data visualization developer creating matplotlib charts.

## TASK
Generate Python code to create a chart based on the provided configuration.

## AVAILABLE IN SCOPE
- data_df: pandas DataFrame with the data (already loaded, do NOT create it)
- config: Dictionary with chart configuration (already defined, do NOT create it)
- pd: pandas module (already imported)
- plt: matplotlib.pyplot module (already imported)

## CHART CONFIGURATION
{config}

## CODE TEMPLATE (common boilerplate)
Every chart follows this structure:
```python
import matplotlib.pyplot as plt
import pandas as pd
[import numpy as np if multi-series]

# Extract data
x_data = data_df[config['x_axis']]
y_data = data_df[config['y_axis']]  # single series
# OR: iterate over config['y_axes']  # wide multi-series
# OR: filter by config['group_by']   # long multi-series

# Create chart
plt.figure(figsize=(10, 6))
[chart-specific code]

# Labels and display
plt.xlabel(config['x_axis'])
plt.ylabel(config['y_axis'] or 'Value')
plt.title(config['title'])
plt.xticks(rotation=45, ha='right')  # prevent label overlap
plt.grid(True, axis='y', alpha=0.3)  # optional
plt.tight_layout()
plt.show()
```

## KEY REQUIREMENTS
1. Check whether config has 'y_axes' (list), 'group_by' (string), or 'y_axis' (string) and handle accordingly:
   - If config has 'y_axes': data is in WIDE FORMAT — produce a GROUPED BAR chart, one bar series per column in y_axes
   - If config has 'group_by': data is in LONG FORMAT — produce a GROUPED BAR chart by filtering data_df by each unique value of config['group_by'], using config['y_axis'] as the metric column. Use sorted unique values as series labels.
   - If config has 'y_axis' only: single series, use data_df[config['y_axis']] directly
2. Create the appropriate chart type using config['chart_type'] (bar, line, scatter, area)
3. Add axis labels, title, legend (when multiple series), and grid
4. Call plt.tight_layout() and plt.show()

## CRITICAL: X-AXIS LABEL OVERLAP PREVENTION
**ALWAYS check and prevent x-axis label overlapping:**
- For categorical data with many categories (>10): rotate labels 45° or 90° AND use ha='right'
- For long text labels: ALWAYS rotate even if few labels
- For dates: rotate 45° with ha='right'
- If labels are still crowded: reduce font size with fontsize=8 or increase figure width

## EXAMPLES

Example 1 - Single series bar chart:
    config = {{"chart_type": "bar", "x_axis": "Product", "y_axis": "Sales", "title": "Sales by Product"}}
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

Example 2 - Wide-format grouped bar (config has 'y_axes'):
    config = {{"chart_type": "bar", "x_axis": "Product_Class", "y_axes": ["Avg_Revenue_Promo", "Avg_Revenue_Non_Promo"], "title": "Promo vs Non-Promo Revenue by Product Class"}}
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

Example 3 - Long-format grouped bar (config has 'group_by'):
    config = {{"chart_type": "bar", "x_axis": "region", "y_axis": "avg_monthly_revenue", "group_by": "year", "title": "Avg Monthly Revenue by Region: 2022 vs 2023"}}
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

    def __init__(self):
        super().__init__()

    @property
    def evaluator(self) -> Evaluator:
        return VisualizerEvaluator()

    def core(self, state: State, llm: LLM) -> State:
        last_retriever_answer = state.get_last_answer(AgentType.RETRIEVER)
        if (
            last_retriever_answer is None
            or "data_df" not in last_retriever_answer.agent_output
            or "data_str" not in last_retriever_answer.agent_output
        ):
            logger.error(
                f"Missing dependencies for visualization from retriever output: last_ret_answer:{last_retriever_answer}, last_retriever_output:{last_retriever_answer.agent_output}"
            )
            raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)

        data_df = last_retriever_answer.agent_output["data_df"]
        data_text = last_retriever_answer.agent_output["data_str"]

        # Extract chart configuration
        formatted_prompt = Visualizer._CHART_CONFIGURATION_PROMPT.format(
            prompt=state.prompt, data=data_text
        )
        response = llm.invoke(formatted_prompt)

        _FALLBACK_CHART_CONFIG = {
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "title": "Chart",
        }
        chart_config = response.extract_json() or _FALLBACK_CHART_CONFIG
        logger.info(f"Chart config : {chart_config}")
        logprobs_chart_config = response.logprobs

        # Generate chart code
        formatted_prompt = Visualizer._CREATE_CHART_PROMPT.format(config=chart_config)
        response = llm.invoke(formatted_prompt)
        code = response.extract_python()
        logger.info(f"Code : {code}")
        logprobs_code = response.logprobs

        # --- Validate by executing in a headless namespace (no display) ---
        exec_code = (
            "import matplotlib.pyplot as plt; plt.switch_backend('Agg')\n"
            + code.replace("plt.show()", "plt.close('all')")
        )
        namespace: dict = {"data_df": data_df, "config": chart_config}
        try:
            exec(exec_code, namespace)  # noqa: S102
            exec_error = ""
        except Exception as e:  # noqa: BLE001 - exec() can raise arbitrary user exceptions
            logger.warning(f"Failed code execution : {type(e).__name__} : {e}")
            exec_error = f"{type(e).__name__}: {e}"

        if exec_error:
            return self.answer(
                state,
                message="The generated code couldn't be executed",
                output={"code": code, "chart_config": chart_config},
                logprobs=logprobs_code + logprobs_chart_config,
                error=exec_error,
            )
        else:
            return self.answer(
                state,
                message="Visualization generated",
                output={"code": code, "chart_config": chart_config},
                logprobs=logprobs_code + logprobs_chart_config,
            )


__all__ = ["Visualizer"]
