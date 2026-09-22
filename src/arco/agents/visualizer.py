from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from arco.core import Agent, AgentException
from arco.evaluators import VisualizerEvaluator

if TYPE_CHECKING:
    from arco.core import LLM, Evaluator, State

logger = logging.getLogger(__name__)


def _store_chart_image(data_df, chart_config: dict, code: str, state: State) -> Path:
    """Execute generated chart code and persist the resulting PNG.

    Artifact persistence belongs to the visualizer rather than the CLI so
    workflows can be executed through other frontends without losing charts.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    buffer = io.BytesIO()
    namespace = {
        "data_df": data_df,
        "config": chart_config,
        "plt": plt,
        "pd": pd,
        "np": np,
        "buf": buffer,
    }
    render_code = code.replace(
        "plt.show()",
        "plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close('all')",
    )
    exec(render_code, namespace)  # noqa: S102 - generated visualization code

    buffer.seek(0)
    image_data = buffer.getvalue()
    if not image_data:
        raise RuntimeError("The generated chart did not produce an image")

    output_dir = Path(state.save_dir) / "storage" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    title = chart_config.get("title", "chart") if chart_config else "chart"
    filename_source = f"{state.run_id}_{title}"
    safe_filename = "".join(
        character if character.isalnum() or character in " _-" else "_"
        for character in filename_source
    ).strip()
    image_path = output_dir / f"{safe_filename or 'chart'}.png"
    image_path.write_bytes(image_data)
    return image_path


class Visualizer(Agent):
    _CHART_CONFIGURATION_PROMPT = """You are a data visualization expert designing chart configurations.

## TASK
Create a JSON configuration object for visualizing the provided data according to the user prompt.

## USER PROMPT
{prompt}

## AVAILABLE DATA COLUMNS
{columns}

## REQUIRED JSON KEYS
- chart_type: One of [bar, line, area, scatter]
- x_axis: Column name for X-axis (string)
- y_axis: Column name for Y-axis (string). Use this for single-series charts.
- y_axes: Use this INSTEAD of y_axis when the data has one column per series (wide format). A list of strings.
- group_by: (OPTIONAL) Use this together with y_axis for long-format grouped charts. A discriminator column name.
- title: Descriptive chart title (string)

Include only keys that are relevant: either y_axis, y_axes, or optionally y_axis + group_by. Never include both y_axis and y_axes.

## OUTPUT FORMAT
Return ONLY a valid JSON object. No markdown. No code fences. No backticks. No explanations.
"""

    @staticmethod
    def _format_chart_spec(config: dict) -> str:
        lines = [
            f"- Chart type: {config.get('chart_type', 'bar')}",
            f'- X-axis column: "{config.get("x_axis", "")}"',
            f"- Dataframe columns : {config.get('df_columns', '')}",
        ]
        if "y_axes" in config:
            cols = "[" + ", ".join(f'"{c}"' for c in config["y_axes"]) + "]"
            lines.append(
                f"- Y-axis columns: {cols} (wide format — one column per series)"
            )
        else:
            lines.append(f'- Y-axis column: "{config.get("y_axis", "")}"')
            if "group_by" in config:
                lines.append(
                    f'- Group by column: "{config["group_by"]}" (long format — filter by unique values)'
                )
        lines.append(f'- Title: "{config.get("title", "Chart")}"')
        return "\n".join(lines)

    _CREATE_CHART_PROMPT = """You are a Python data visualization developer creating matplotlib charts.

## TASK
Generate Python code to create a chart according to the specification below.

## AVAILABLE IN SCOPE
- data_df: pandas DataFrame with the data (already loaded, do NOT create it)
- pd: pandas module (already imported)
- plt: matplotlib.pyplot module (already imported)

## CHART SPECIFICATION
{spec}

## NOTES
- Use the EXACT column names from the specification as string literals in your code (e.g., data_df["column_name"]). Do NOT read column names from a config dict — hardcode them.
- The code must be self-contained and static: no if/else branching on chart structure, no dynamic dict lookups.
- For single-series charts: access the x and y columns directly on the DataFrame.
- For wide format (multiple y-axis columns listed): produce a grouped bar chart by iterating over the y columns.
- For long format (has group_by): filter data_df by each unique value of the group_by column.

## REQUIREMENTS
1. Use the correct chart type (bar, line, scatter, or area)
2. Add axis labels, title, legend (when multiple series), and grid
3. Call plt.tight_layout() and plt.show()
4. Make sure that, when accessing the Dataframe, the columns used are specified in the specification of dataframe columns. They may not match exactly the names provided by X-axis, Y-axis, Y-axes or group_by

## EXAMPLE
Specification:
- Chart type: bar
- X-axis column: "Product"
- Y-axis column: "Sales"
- Title: "Sales by Product"
- Dataframe columns : ["Product", "Sales"]

Code:
import matplotlib.pyplot as plt
import pandas as pd

x_data = data_df["Product"]
y_data = data_df["Sales"]

plt.figure(figsize=(10, 6))
plt.bar(x_data, y_data)
plt.xlabel("Product")
plt.ylabel("Sales")
plt.title("Sales by Product")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
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
        last_retriever_answer = state.get_last_answer("Retriever")
        if (
            last_retriever_answer is None
            or "data_df" not in last_retriever_answer.agent_output
            or "data_str" not in last_retriever_answer.agent_output
        ):
            logger.error(
                f"Missing dependencies for visualization from retriever output: last_ret_answer:{last_retriever_answer}, last_retriever_output:{last_retriever_answer.agent_output}"
            )
            raise AgentException(missing_answer_from_type="Retriever")

        data_df = last_retriever_answer.agent_output["data_df"]

        # Extract chart configuration
        data_columns = ", ".join(str(c) for c in data_df.columns)
        formatted_prompt = Visualizer._CHART_CONFIGURATION_PROMPT.format(
            prompt=state.prompt, columns=data_columns
        )
        response = llm.invoke(formatted_prompt)

        _FALLBACK_CHART_CONFIG = {
            "chart_type": "line",
            "x_axis": str(data_df.columns[0]) if len(data_df.columns) > 0 else "date",
            "y_axis": str(data_df.columns[1]) if len(data_df.columns) > 1 else "value",
            "title": "Chart",
        }
        chart_config = response.extract_json() or _FALLBACK_CHART_CONFIG
        chart_config.update({"df_columns": data_columns})
        logger.info(f"Chart config : {chart_config}")
        logprobs_chart_config = response.logprobs

        # Generate chart code
        chart_spec = Visualizer._format_chart_spec(chart_config)
        formatted_prompt = Visualizer._CREATE_CHART_PROMPT.format(spec=chart_spec)
        response = llm.invoke(formatted_prompt)
        code = response.extract_python()
        logger.info(f"Code : {code}")
        logprobs_code = response.logprobs

        # --- Validate by executing in a headless namespace (no display) ---
        exec_code = (
            "import pandas as pd; import matplotlib.pyplot as plt; plt.switch_backend('Agg')\n"
            + code.replace("plt.show()", "plt.close('all')")
        )
        namespace: dict = {"data_df": data_df}
        try:
            exec(exec_code, namespace)  # noqa: S102
            exec_error = ""
        except Exception as e:  # noqa: BLE001 - exec() can raise arbitrary user exceptions
            logger.warning(f"Failed code execution : {type(e).__name__} : {e}")
            exec_error = f"{type(e).__name__}: {e}"

        chart_path: Path | None = None
        if not exec_error and state.enable_storage:
            try:
                chart_path = _store_chart_image(
                    data_df=data_df,
                    chart_config=chart_config,
                    code=code,
                    state=state,
                )
                logger.info(f"Chart stored at: {chart_path}")
            except Exception as e:  # noqa: BLE001 - generated code can fail arbitrarily
                logger.warning(f"Failed to store chart : {type(e).__name__} : {e}")
                exec_error = f"{type(e).__name__}: {e}"

        output = {"code": code, "chart_config": chart_config}
        if chart_path is not None:
            output["image_path"] = str(chart_path)

        if exec_error:
            return self.answer(
                state,
                message="The generated visualization couldn't be completed",
                output=output,
                logprobs=logprobs_code + logprobs_chart_config,
                error=exec_error,
                thinking=response.reasoning,
            )

        return self.answer(
            state,
            message="Visualization generated",
            output=output,
            logprobs=logprobs_code + logprobs_chart_config,
            thinking=response.reasoning,
        )


__all__ = ["Visualizer"]
