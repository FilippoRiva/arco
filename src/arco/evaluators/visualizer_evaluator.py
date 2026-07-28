import json
from typing import TYPE_CHECKING, Any

from arco.core import AgentType, Evaluation, Evaluator, LLMAnswer, llm_tools
from arco.core.llm_tools import (
    compute_weighted_score,
    fill_json_schema,
)

if TYPE_CHECKING:
    from arco.core import Answer, State

import logging

logger = logging.getLogger(__name__)

_NO_GT_SCHEMA: dict[str, dict[str, Any]] = {
    "data_suitability": {"score": 1, "reasoning": "Missing"},
    "axis_mapping": {"score": 1, "reasoning": "Missing", "columns_exist": False},
    "code_quality": {"score": 1, "reasoning": "Missing", "would_render": False},
    "goal_alignment": {"score": 1, "reasoning": "Missing"},
}

_GT_SCHEMA: dict[str, dict[str, Any]] = {
    "axis_correctness": {
        "score": 1,
        "reasoning": "Missing",
        "x_match": False,
        "y_match": False,
    },
    "chart_type": {"score": 1, "reasoning": "Missing", "type_match": False},
    "functional_equivalence": {
        "score": 1,
        "reasoning": "Missing",
        "would_render": False,
    },
    "explicit_requirements": {
        "score": 5,
        "reasoning": "Missing - default N/A",
        "violations": [],
    },
}

_NO_GT_WEIGHTS: dict[str, float] = {
    "data_suitability": 0.30,
    "axis_mapping": 0.30,
    "code_quality": 0.20,
    "goal_alignment": 0.20,
}

_GT_WEIGHTS: dict[str, float] = {
    "axis_correctness": 0.20,
    "chart_type": 0.15,
    "functional_equivalence": 0.55,
    "explicit_requirements": 0.10,
}


class VisualizerEvaluator(Evaluator):
    JUDGE_PROMPT = """You are an expert data visualization evaluator. Assess the quality of a generated visualization based on the data and the user's goal. There is NO reference visualization — evaluate standalone quality.

## USER PROMPT
{prompt}

## AVAILABLE DATA
Columns: {data_columns}
Sample rows:
{data_sample}

## GENERATED OUTPUT
Chart Configuration:
{gen_config}

Chart Code:
```python
{gen_code}
```

## EVALUATION CRITERIA

Rate each criterion on a scale of 1-5:

### 1. DATA SUITABILITY
Is the chart type appropriate for the data structure?
- Consider the data: categorical → bar/column, time-series → line, correlation → scatter, cumulative → area
- A chart type that *works* for the data should score well even if it's not the absolute textbook choice
[1=Wrong chart type for data, 3=Acceptable/works fine, 5=Excellent choice]

### 2. AXIS MAPPING
Are the X and Y axes using appropriate columns from the data?
- The config may have 'y_axis' (single column), 'y_axes' (list), or 'y_axis'+'group_by' — all are valid approaches
- Accept reasonable column-name variations (e.g., 'date' vs 'sold_date', 'val' vs 'total_value') — exact string match is not required
- Are the axes semantically correct (e.g., time on X, measure on Y)?
[1=Wrong/missing columns, 3=Acceptable mapping with minor name differences or missing a series, 5=Perfect mapping with all required series]

### 3. CODE QUALITY
Will the matplotlib code execute correctly and produce a readable chart?
- Syntactically correct Python/matplotlib
- Proper data references, labels, and formatting
- Would plt.show() produce a clean output?
[1=Would fail/unreadable, 3=Minor issues, 5=Clean and correct]

### 4. GOAL ALIGNMENT
Does the visualization effectively address the user's goal?
- Does it show the right information to answer the user's question?
- Is the title/labeling informative?
[1=Misses the goal, 3=Partially addresses it, 5=Fully addresses the goal]

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "data_suitability": {{"score": <1-5>, "reasoning": "<brief>"}},
  "axis_mapping": {{"score": <1-5>, "reasoning": "<brief>", "columns_exist": <true/false>}},
  "code_quality": {{"score": <1-5>, "reasoning": "<brief>", "would_render": <true/false>}},
  "goal_alignment": {{"score": <1-5>, "reasoning": "<brief>"}}
}}"""

    GT_JUDGE_PROMPT = """You are an expert data visualization evaluator. Your task is to assess whether a generated visualization achieves the same analytical purpose as a reference visualization.

## REFERENCE (GROUND TRUTH)
Chart Configuration:
{gt_config}

Chart Code:
```python
{gt_code}
```

## GENERATED OUTPUT
Chart Configuration:
{gen_config}

Chart Code:
```python
{gen_code}
```

## EXPLICIT USER REQUIREMENTS
{explicit_requirements}

## EVALUATION CRITERIA

Rate each criterion on a scale of 1-5:

### 1. AXIS CORRECTNESS
Do the X and Y axes convey the same information as the reference?
- Column names need not match exactly — focus on whether the *same data columns* are being visualised (e.g., 'date' vs 'sold_date' is fine)
- Swapped axes are acceptable if they produce a readable chart (e.g., horizontal bar intentionally swaps X and Y)
- Configs may use 'y_axis', 'y_axes', or 'y_axis'+'group_by' — focus on which columns end up on each axis, not the exact key name
[1=Completely different columns, 3=Mostly same columns with minor differences, 5=Same information conveyed]

### 2. CHART TYPE CORRECTNESS
Does the chart type serve the same visual purpose as the reference?
- Exact type match is not required — a column chart and a bar chart are functionally the same
- Consider whether the chosen type can display the same information effectively
[1=Type cannot convey the same information, 3=Different but reasonable alternative, 5=Same or functionally identical type]

### 3. FUNCTIONAL EQUIVALENCE
Would the generated code produce a visually equivalent chart?
- Ignore import statements, variable naming, and code style differences
- Ignore cosmetic differences (colour, grid style, font size)
- Focus on: Would a viewer draw the same conclusion from both charts?
[1=Would produce a completely different visual, 3=Minor visual differences, 5=Visually equivalent]

### 4. EXPLICIT REQUIREMENTS COMPLIANCE
ONLY evaluate requirements that are non-null in EXPLICIT USER REQUIREMENTS.
For each non-null requirement, check if the generated code complies.
If all explicit requirements are null, give score of 5 (not applicable).
[1=Major violations, 3=Partial compliance, 5=Full compliance or N/A]

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "axis_correctness": {{"score": <1-5>, "reasoning": "<brief>", "x_match": <true/false>, "y_match": <true/false>}},
  "chart_type": {{"score": <1-5>, "reasoning": "<brief>", "type_match": <true/false>}},
  "functional_equivalence": {{"score": <1-5>, "reasoning": "<brief>", "would_render": <true/false>}},
  "explicit_requirements": {{"score": <1-5>, "reasoning": "<brief>", "violations": []}}
}}"""

    def _eval(self, state: State, judge_provider: str, judge_model: str):
        """
        Uses an LLM judge to score chart quality based on data suitability,
        axis mapping, code quality, and goal alignment.
        """
        llm = llm_tools.get_llm(provider=judge_provider, model=judge_model)

        last_visualizer_answer: Answer = state.get_last_answer(AgentType.VISUALIZER)
        last_retriever_answer: Answer = state.get_last_answer(AgentType.RETRIEVER)
        data_df = last_retriever_answer.agent_output["data_df"]

        if data_df is not None and hasattr(data_df, "columns"):
            data_columns = list(data_df.columns)
            data_sample = data_df.head(5).to_string(index=False)
        else:
            data_text = last_retriever_answer.agent_output["data_str"]
            data_columns = []
            data_sample = data_text[:500] if data_text else ""

        max_code_len = 2000
        code: str = last_visualizer_answer.agent_output["code"]
        gen_code_truncated = code[:max_code_len] if len(code) > max_code_len else code

        formatted_prompt = VisualizerEvaluator.JUDGE_PROMPT.format(
            prompt=state.prompt,
            data_columns=", ".join(data_columns),
            data_sample=data_sample[:1500],
            gen_config=json.dumps(
                last_visualizer_answer.agent_output["chart_config"], indent=2
            ),
            gen_code=gen_code_truncated,
        )

        response = llm.invoke(formatted_prompt)

        evaluation_dict = fill_json_schema(response.extract_json(), _NO_GT_SCHEMA)
        overall_score = compute_weighted_score(evaluation_dict, _NO_GT_WEIGHTS)
        last_visualizer_answer.evaluation = Evaluation(score=overall_score)

    def _batch_eval(self, states: list[State]) -> bool:
        return False

    def _gt_eval(self, answer: Answer, gt_data, judge_provider: str, judge_model: str):
        """
        Evaluate visualization quality using LLM-as-a-Judge.

        Args:
            answer: The state to be evaluated.
            llm: the LLM used for LLM-as-a-Judge inference.
            gt_config: Expected chart configuration dict.
            gt_code: Expected chart code string.
            gt_visual_requirements: Optional dict of explicit styling requirements.
        """
        llm = llm_tools.get_llm(provider=judge_provider, model=judge_model)
        gt_config = gt_data["chart_config"]
        gt_code = gt_data["chart_code"]
        gt_visual_requirements = gt_data["visual_requirements"]
        if gt_visual_requirements:
            req_display = "\n".join(
                [
                    f"- {k}: {v}"
                    if v is not None
                    else f"- {k}: (not specified - ignore)"
                    for k, v in gt_visual_requirements.items()
                ]
            )
        else:
            req_display = "None specified - ignore all styling requirements"

        code: str = answer.agent_output["code"]
        if code is None:
            answer.gt_evaluation = Evaluation(score=0)
            return

        # Truncate code if too long
        max_code_len = 2000
        gen_code_truncated = code[:max_code_len] if len(code) > max_code_len else code
        gt_code_truncated = (
            gt_code[:max_code_len] if len(gt_code) > max_code_len else gt_code
        )

        # Format the judge prompt
        formatted_prompt = VisualizerEvaluator.GT_JUDGE_PROMPT.format(
            gt_config=json.dumps(gt_config, indent=2),
            gt_code=gt_code_truncated,
            gen_config=json.dumps(answer.agent_output["chart_config"], indent=2),
            gen_code=gen_code_truncated,
            explicit_requirements=req_display,
        )

        # Get judgment
        response: LLMAnswer = llm.invoke(formatted_prompt)

        # Parse JSON response
        evaluation_dict = fill_json_schema(response.extract_json(), _GT_SCHEMA)

        # Compute overall score
        overall_score = compute_weighted_score(evaluation_dict, _GT_WEIGHTS)
        answer.gt_evaluation = Evaluation(score=overall_score)
        logger.debug(f"Evaluation successful : score={overall_score}")
        return


__all__ = ["VisualizerEvaluator"]
