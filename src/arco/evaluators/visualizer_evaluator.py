import json
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco.core import AgentType, Answer, Evaluation, Evaluator, llm_tools

if TYPE_CHECKING:
    from arco.core import State


class VisualizerEvaluator(Evaluator):
    VIS_JUDGE_NO_GT_PROMPT = """You are an expert data visualization evaluator. Assess the quality of a generated visualization based on the data and the user's goal. There is NO reference visualization — evaluate standalone quality.

    ## VISUALIZATION GOAL
    {visualization_goal}

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
    - Bar/column for categorical comparisons, line for time-series trends, scatter for correlations, area for cumulative values
    - Does the data have enough points/categories for this chart type?
    [1=Wrong chart type for data, 3=Acceptable, 5=Ideal choice]

    ### 2. AXIS MAPPING
    Are the X and Y axes using appropriate columns from the data?
    - The config may have 'y_axis' (single column), 'y_axes' (list of columns for wide-format multi-series), or 'y_axis'+'group_by' (long-format multi-series where series are filtered by a discriminator column). All are valid.
    - Do the column names in the config actually exist in the data? For 'y_axes', each listed column must exist. For 'y_axis'+'group_by', both y_axis and group_by must exist as actual data columns.
    - Are the axes semantically correct (e.g., time on X, measure on Y)?
    - For comparison goals (A vs B for different years/categories): a single y_axis with group_by pointing to the discriminator column is correct; y_axes with columns that DON'T exist in data should score low.
    [1=Wrong/missing columns, 3=Acceptable mapping or missing one series, 5=Perfect mapping with all required series]

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

    @staticmethod
    def _parse_vis_no_gt_judge_json(raw_text: str) -> dict:
        """Parse no-GT visualization judge JSON response."""
        try:
            content = raw_text.strip().replace("```json", "").replace("```", "").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

            start = content.find("{")
            end = content.rfind("}")

            if start != -1 and end != -1:
                parsed = json.loads(content[start : end + 1])
                for criterion in [
                    "data_suitability",
                    "axis_mapping",
                    "code_quality",
                    "goal_alignment",
                ]:
                    if criterion not in parsed:
                        parsed[criterion] = {"score": 1, "reasoning": "Missing"}
                return parsed
        except Exception as _:
            return {
                "data_suitability": {"score": 1, "reasoning": "Parse failed"},
                "axis_mapping": {
                    "score": 1,
                    "reasoning": "Parse failed",
                    "columns_exist": False,
                },
                "code_quality": {
                    "score": 1,
                    "reasoning": "Parse failed",
                    "would_render": False,
                },
                "goal_alignment": {"score": 1, "reasoning": "Parse failed"},
            }

    @staticmethod
    def _compute_vis_no_gt_score(evaluation: dict) -> float:
        """Compute weighted normalized score from no-GT vis judge evaluation."""
        weights = {
            "data_suitability": 0.30,
            "axis_mapping": 0.30,
            "code_quality": 0.20,
            "goal_alignment": 0.20,
        }
        total = 0.0
        for criterion, weight in weights.items():
            raw_score = evaluation.get(criterion, {}).get("score", 1)
            normalized = (raw_score - 1) / 4.0
            total += normalized * weight
        return total

    @staticmethod
    def judge(state: State, llm: BaseChatModel):
        """Evaluate visualization quality without ground truth using LLM-as-a-Judge."""

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

        formatted_prompt = VisualizerEvaluator.VIS_JUDGE_NO_GT_PROMPT.format(
            visualization_goal=state.prompt,
            data_columns=", ".join(data_columns),
            data_sample=data_sample[:1500],
            gen_config=json.dumps(
                last_visualizer_answer.agent_output["chart_config"], indent=2
            ),
            gen_code=gen_code_truncated,
        )

        response = llm.invoke(formatted_prompt)
        raw_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        evaluation = VisualizerEvaluator._parse_vis_no_gt_judge_json(raw_content)
        overall_score = VisualizerEvaluator._compute_vis_no_gt_score(evaluation)
        last_visualizer_answer.evaluation = Evaluation(score=overall_score)

    def _eval(self, state: State, judge_provider: str, judge_model: str):
        """
        Uses an LLM judge to score chart quality based on data suitability,
        axis mapping, code quality, and goal alignment.
        """
        llm = llm_tools.get_llm(provider=judge_provider, model=judge_model)
        VisualizerEvaluator.judge(state, llm)

    VIS_JUDGE_PROMPT_GT = """You are an expert data visualization evaluator. Your task is to assess whether a generated visualization achieves the same analytical purpose as a reference visualization.
    
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
    Do X and Y axes use the SAME data columns as the reference?
    - Column names must match exactly (case-insensitive)
    - Axes cannot be swapped (x must be x, y must be y)
    - Configs may use 'y_axis' (single column), 'y_axes' (list of columns for wide-format multi-series), or 'y_axis'+'group_by' (long-format multi-series). These are all valid multi-series approaches. If the reference uses 'group_by' and the generated uses 'y_axes' (or vice versa), focus on whether the SAME columns are ultimately visualized — not on the exact key name.
    [1=Wrong columns, 3=Partial match, 5=Exact match]

    ### 2. CHART TYPE CORRECTNESS
    Is the chart type the same as the reference?
    - line, bar, scatter, area must match exactly
    - Variations within type are acceptable (e.g., grouped bar vs stacked bar)
    [1=Wrong type, 3=Similar type, 5=Exact match]

    ### 3. FUNCTIONAL EQUIVALENCE
    Would the generated code produce a visually equivalent chart?
    - Ignore import statements and variable naming
    - Ignore code style/formatting differences
    - Focus on: Will plt.show() produce the same visual output?
    [1=Would fail/wrong output, 3=Minor visual differences, 5=Equivalent output]

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

    @staticmethod
    def _parse_vis_judge_json(raw_text: str) -> dict:
        """Parse visualization judge JSON response with robust error handling."""
        try:
            content = raw_text.strip().replace("```json", "").replace("```", "").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

            start = content.find("{")
            end = content.rfind("}")

            if start != -1 and end != -1:
                parsed = json.loads(content[start : end + 1])

                # Ensure all criteria exist
                for criterion in [
                    "axis_correctness",
                    "chart_type",
                    "functional_equivalence",
                    "explicit_requirements",
                ]:
                    if criterion not in parsed:
                        parsed[criterion] = {
                            "score": 1,
                            "reasoning": "Missing",
                            "violations": [],
                        }

                return parsed
        except Exception:
            return {
                "axis_correctness": {
                    "score": 1,
                    "reasoning": "Parse failed",
                    "x_match": False,
                    "y_match": False,
                },
                "chart_type": {
                    "score": 1,
                    "reasoning": "Parse failed",
                    "type_match": False,
                },
                "functional_equivalence": {
                    "score": 1,
                    "reasoning": "Parse failed",
                    "would_render": False,
                },
                "explicit_requirements": {
                    "score": 5,
                    "reasoning": "Parse failed - default N/A",
                    "violations": [],
                },
            }

    @staticmethod
    def _compute_visualization_score(evaluation: dict) -> float:
        """Compute weighted normalized score from judge evaluation.

        Returns:
            Score between 0.0 and 1.0
        """
        weights = {
            "axis_correctness": 0.40,
            "chart_type": 0.30,
            "functional_equivalence": 0.20,
            "explicit_requirements": 0.10,
        }

        total_score = 0.0
        for criterion, weight in weights.items():
            raw_score = evaluation.get(criterion, {}).get("score", 1)
            # Normalize from 1-5 scale to 0-1
            normalized = (raw_score - 1) / 4.0
            total_score += normalized * weight

        return round(total_score, 6)

    @staticmethod
    def judge_from_ground_truth(
        answer: Answer,
        llm: BaseChatModel,
        gt_config: str = None,
        gt_code: str = None,
        gt_visual_requirements: dict = None,
    ) -> State:
        """
        Evaluate visualization quality using LLM-as-a-Judge.

        Args:
            answer: The state to be evaluated.
            llm: the BaseChatModel used for LLM-as-a-Judge inference.
            gt_config: Expected chart configuration dict.
            gt_code: Expected chart code string.
            gt_visual_requirements: Optional dict of explicit styling requirements.
        """

        # Format explicit requirements for display
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

        if not gt_code:
            raise Exception("gt_code cannot be None")

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
        formatted_prompt = VisualizerEvaluator.VIS_JUDGE_PROMPT_GT.format(
            gt_config=json.dumps(gt_config, indent=2),
            gt_code=gt_code_truncated,
            gen_config=json.dumps(answer.agent_output["chart_config"], indent=2),
            gen_code=gen_code_truncated,
            explicit_requirements=req_display,
        )

        # Get judgment
        response = llm.invoke(formatted_prompt)
        raw_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse JSON response
        evaluation_dict = VisualizerEvaluator._parse_vis_judge_json(raw_content)

        # Compute overall score
        overall_score = VisualizerEvaluator._compute_visualization_score(
            evaluation_dict
        )
        answer.gt_evaluation = Evaluation(score=overall_score)
        return

    def _gt_eval(self, answer: Answer, gt_data, judge_provider: str, judge_model: str):
        llm = llm_tools.get_llm(provider=judge_provider, model=judge_model)
        VisualizerEvaluator.judge_from_ground_truth(
            answer,
            llm=llm,
            gt_config=gt_data["chart_config"],
            gt_code=gt_data["chart_code"],
            gt_visual_requirements=gt_data["visual_requirements"],
        )
