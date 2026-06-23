import json
from copy import deepcopy
from json import JSONDecodeError
from typing import Optional, Dict, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco import llm_tools
from arco import tracing
from arco.core import Agent, Answer, AgentType
from arco.core.agent import AgentException
from arco.evaluators import VisualizerEvaluator
from arco.llm_tools import CoTRefiner

if TYPE_CHECKING:
    from arco.core import Evaluator, AgentConfig, State
    from arco.tracing import TracingHelper


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

    def __init__(self, trace_helper: TracingHelper):
        super().__init__(trace_helper)
        self.type = AgentType.VISUALIZER

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
    def _extract_chart_config(state: State, llm: BaseChatModel | CoTRefiner,
                              trace_helper: Optional[TracingHelper] = None) -> tuple[
        dict[str, str], list[float | int] | None]:
        """Infer a compact chart configuration from the looked-up data.

        Prompts the LLM to return a minified JSON config and parses it into a
        Python dict. Data is NOT included in the config (it's passed separately as DataFrame).

        Args:
            state: Conversation state; should include 'data' and optionally 'visualization_goal'.
            llm: ChatOllama instance used to infer the chart configuration.
            trace_helper : Optional tracing Helper for Phoenix integration

        Returns:
            A proposal chart_configuration
        """
        ans_to_check = state.get_last_answer(AgentType.RETRIEVER)
        if ans_to_check is None:
            raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)
        last_retriever_answer: Answer = ans_to_check
        if last_retriever_answer.data_str is None:
            raise AgentException(missing_dataframe_from_type=AgentType.RETRIEVER)
        data_text = last_retriever_answer.data_str

        helper = trace_helper or TracingHelper()
        visualization_goal = state.visualization_goal or state.prompt
        with helper.start_span(
                "chart_config_extraction",
                kind="tool",
                input_data={
                    "visualization_goal": tracing.truncate_trace_text(visualization_goal),
                    "data_preview": tracing.truncate_trace_text(data_text),
                },
        ) as span:
            formatted_prompt = Visualizer._CHART_CONFIGURATION_PROMPT.format(
                data=data_text, visualization_goal=visualization_goal
            )
            response = llm.invoke(formatted_prompt)
            logprobs = llm_tools.extract_logprobs(response)
            raw: str = str(response.content) if hasattr(response, "content") else str(response)
            chart_config = Visualizer._parse_chart_config(raw)
            tracing.set_output(
                span,
                {
                    "raw_response": tracing.truncate_trace_text(raw),
                    "chart_config": chart_config,
                },
            )
            # Do NOT include data in chart_config - it will be passed separately as DataFrame
            # print("This is the chart_config: " + str(chart_config))
            return chart_config, logprobs

    @staticmethod
    def _create_chart(chart_config: dict, llm: BaseChatModel | CoTRefiner,
                      trace_helper: Optional[TracingHelper] = None) -> tuple[str, list[float | int] | None]:
        """Ask the LLM to emit matplotlib code for the given chart configuration.

        Args:
            llm: ChatOllama instance used to generate the plotting code.
            trace_helper : Optional tracing Helper for Phoenix integration

        Returns:
            A Python code string (without Markdown fences) that, when executed,
            renders the chart using matplotlib.
        """
        helper = trace_helper or TracingHelper()
        with helper.start_span(
                "chart_code_generation",
                kind="tool",
                input_data={"chart_config": chart_config},
        ) as span:
            formatted_prompt = Visualizer._CREATE_CHART_PROMPT.format(config=chart_config)
            response = llm.invoke(formatted_prompt)
            logprobs = llm_tools.extract_logprobs(response)
            code: str = str(response.content) if hasattr(response, "content") else str(response)
            cleaned_code = code.replace("```python", "").replace("```", "").strip()
            tracing.set_output(span, {"code": tracing.truncate_trace_text(cleaned_code)})
            # clean any accidental fences
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
        try:
            ans_to_check = state.get_last_answer(AgentType.RETRIEVER)
            if ans_to_check is None:
                raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)
            last_retriever_answer: Answer = ans_to_check

            data_df = last_retriever_answer.data_df
            # if data_df is not None:
            #    print(f"Using DataFrame with shape: {data_df.shape}, columns: {list(data_df.columns)}")
            # else:
            #    print("Warning: No DataFrame available in state")

            # Extract chart configuration
            chart_config, logprobs_chart_config = Visualizer._extract_chart_config(state, llm,
                                                                                   trace_helper=self.trace_helper)

            # Generate chart code
            code, logprobs_code = Visualizer._create_chart(chart_config=chart_config, llm=llm,
                                                           trace_helper=self.trace_helper)

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
                with self.trace_helper.start_span(
                        "visualization_validation",
                        kind="tool",
                        input_data={
                            "chart_config": chart_config,
                            "dataframe": tracing.summarize_dataframe(data_df),
                            "code": tracing.truncate_trace_text(code),
                        },
                ) as span:
                    exec(exec_code, namespace)  # noqa: S102
                    exec_error = ""
                    tracing.set_output(span, {"validation": "passed"})
            except Exception as e:
                exec_error = f"{type(e).__name__}: {e}"

            if exec_error:
                answer = Answer(
                    agent_id=self.type,
                    message="The generated code couldn't be executed",
                    code=code,
                    chart_config=chart_config,
                    agent_config=deepcopy(state.get_agent_config(self.type)),
                    error=exec_error
                )
            else:
                answer = Answer(
                    agent_id=self.type,
                    message="The code for a proper visualization is:\n\n'''python\n" + code + f"\n'''\n The configuration for matplotlib is:\n{chart_config}",
                    code=code,
                    chart_config=chart_config,
                    agent_config=deepcopy(state.get_agent_config(self.type)),
                    logprobs=logprobs_code + logprobs_chart_config if logprobs_code is not None and logprobs_chart_config is not None else None
                )

            return state.add_answer(answer)
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            # Handle the case where the LLM or logic failed entirely
            answer = Answer(
                agent_id=self.type,
                message="Couldn't create the visualization code",
                error=f"Internal Exception: {str(e)}",
                agent_config=deepcopy(state.get_agent_config(self.type)),
            )
            return state.add_answer(answer)

    def get_evaluator(self, agent_config: AgentConfig) -> Evaluator:
        return VisualizerEvaluator(agent_config=agent_config)

    def can_evaluate_from_gt(self, agent_config: AgentConfig) -> bool:
        if agent_config.gt_config and agent_config.gt_code and agent_config.gt_requirements:
            return True
        return False
