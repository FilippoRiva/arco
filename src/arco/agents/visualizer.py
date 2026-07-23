import json
from copy import deepcopy
from json import JSONDecodeError
from typing import Dict, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco.core import Agent, Answer, AgentType, llm_tools
from arco.core.agent import AgentException
from arco.evaluators import VisualizerEvaluator

if TYPE_CHECKING:
    from arco.core.llm_tools import CoTRefiner
    from arco.core import Evaluator, State


class Visualizer(Agent):
    _CHART_CONFIGURATION_PROMPT = """You are a data visualization expert designing chart configurations.

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

    @staticmethod
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
                return json.loads(text[start: end + 1])
            raise JSONDecodeError  # Falls to default
        except (JSONDecodeError, AttributeError) as _:
            # fallback config
            return {
                "chart_type": "line",
                "x_axis": "date",
                "y_axis": "value",
                "title": "Chart",
            }

    @staticmethod
    def _extract_chart_config(state: State, llm: BaseChatModel | CoTRefiner) \
            -> tuple[dict[str, str], list[float | int] | None]:
        """Infer a compact chart configuration from the looked-up data.

        Prompts the LLM to return a minified JSON config and parses it into a
        Python dict. Data is NOT included in the config (it's passed separately as DataFrame).

        Args:
            state: Conversation state; should include 'data'.
            llm: ChatOllama instance used to infer the chart configuration.

        Returns:
            A proposal chart_configuration
        """
        ans_to_check = state.get_last_answer(AgentType.RETRIEVER)
        if ans_to_check is None:
            raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)
        last_retriever_answer: Answer = ans_to_check
        if last_retriever_answer.agent_output['data_str'] is None:
            raise AgentException(missing_dataframe_from_type=AgentType.RETRIEVER)
        data_text = last_retriever_answer.agent_output['data_str']

        visualization_goal = state.prompt

        formatted_prompt = Visualizer._CHART_CONFIGURATION_PROMPT.format(
            data=data_text, visualization_goal=visualization_goal
        )
        response = llm.invoke(formatted_prompt)
        logprobs = llm_tools.extract_logprobs(response)
        raw: str = str(response.content) if hasattr(response, "content") else str(response)
        chart_config = Visualizer._parse_chart_config(raw)
        return chart_config, logprobs

    @staticmethod
    def _create_chart(chart_config: dict, llm: BaseChatModel | CoTRefiner) -> tuple[str, list[float | int] | None]:
        """Ask the LLM to emit matplotlib code for the given chart configuration.

        Args:
            llm: ChatOllama instance used to generate the plotting code.

        Returns:
            A Python code string (without Markdown fences) that, when executed,
            renders the chart using matplotlib.
        """
        formatted_prompt = Visualizer._CREATE_CHART_PROMPT.format(config=chart_config)
        response = llm.invoke(formatted_prompt)
        logprobs = llm_tools.extract_logprobs(response)
        code: str = str(response.content) if hasattr(response, "content") else str(response)
        cleaned_code = code.replace("```python", "").replace("```", "").strip()
        return cleaned_code, logprobs

    def core(self, state: State, llm: BaseChatModel | CoTRefiner) -> State:
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
        ans_to_check = state.get_last_answer(AgentType.RETRIEVER)
        if ans_to_check is None:
            raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)
        last_retriever_answer: Answer = ans_to_check

        data_df = last_retriever_answer.agent_output['data_df']
        # if data_df is not None:
        #    print(f"Using DataFrame with shape: {data_df.shape}, columns: {list(data_df.columns)}")
        # else:
        #    print("Warning: No DataFrame available in state")

        # Extract chart configuration
        chart_config, logprobs_chart_config = Visualizer._extract_chart_config(state, llm)

        # Generate chart code
        code, logprobs_code = Visualizer._create_chart(chart_config=chart_config, llm=llm)

        # --- Validate by executing in a headless namespace (no display) ---
        # Switch to Agg (non-interactive) backend to avoid tkinter threading
        # issues when running best-of-n from a non-main thread on Windows.
        exec_code = (
                "import matplotlib.pyplot as plt; plt.switch_backend('Agg')\n"
                + code.replace("plt.show()", "plt.close('all')")
        )
        namespace: Dict = {
            "data_df": data_df,
            "config": chart_config
        }
        try:
            exec(exec_code, namespace)  # noqa: S102
            exec_error = ""
        except Exception as e:
            exec_error = f"{type(e).__name__}: {e}"

        if exec_error:
            answer = Answer(
                agent_id=self.type,
                message="The generated code couldn't be executed",
                agent_output={
                    "code": code,
                    "chart_config": chart_config
                },
                agent_config=deepcopy(state.get_agent_config(self.type)),
                error=exec_error
            )
        else:
            answer = Answer(
                agent_id=self.type,
                message="The code for a proper visualization is:\n\n'''python\n" + code + f"\n'''\n The configuration for matplotlib is:\n{chart_config}",
                agent_output={
                    "code": code,
                    "chart_config": chart_config
                },
                agent_config=deepcopy(state.get_agent_config(self.type)),
                logprobs=logprobs_code + logprobs_chart_config if logprobs_code is not None and logprobs_chart_config is not None else None
            )

        return state.add_answer(answer)

    @staticmethod
    def get_evaluator() -> Evaluator:
        return VisualizerEvaluator()
