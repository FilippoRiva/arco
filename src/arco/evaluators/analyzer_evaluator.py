import logging
from typing import Any

from arco.core import Answer, Evaluation, Evaluator, State, get_llm
from arco.core.llm_tools import (
    compute_weighted_score,
    fill_json_schema,
)

logger = logging.getLogger(__name__)

_NO_GT_SCHEMA: dict[str, dict[str, Any]] = {
    "correctness": {"score": 0, "reasoning": "Missing", "issues": []},
    "completeness": {"score": 0, "reasoning": "Missing", "missing": []},
    "faithfulness": {"score": 0, "reasoning": "Missing", "hallucinations": []},
}

_GT_SCHEMA: dict[str, dict[str, Any]] = {
    "factual_accuracy": {"score": 1, "reasoning": "Missing"},
    "coverage": {"score": 1, "reasoning": "Missing"},
}

_NO_GT_WEIGHTS: dict[str, float] = {
    "correctness": 1 / 3,
    "completeness": 1 / 3,
    "faithfulness": 1 / 3,
}

_GT_WEIGHTS: dict[str, float] = {
    "factual_accuracy": 0.5,
    "coverage": 0.5,
}


def _normalize_flat_judgement(parsed: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise the GT judge's flat ``{"factual_accuracy": 5}`` to the nested
    ``{"factual_accuracy": {"score": 5}}`` format expected by *fill_json_schema*."""
    result: dict[str, Any] = {}
    if parsed is None:
        return result
    for key in ("factual_accuracy", "coverage"):
        value = parsed.get(key)
        if isinstance(value, dict):
            result[key] = value
        elif isinstance(value, (int, float)):
            result[key] = {"score": value, "reasoning": parsed.get("reasoning", "")}
    if "reasoning" in parsed:
        for key, value in result.items():
            if "reasoning" not in value:
                value["reasoning"] = parsed["reasoning"]
    return result


class AnalyzerEvaluator(Evaluator):
    ANALYZE_JUDGE_PROMPT_GT = """You are an expert evaluator comparing a generated data analysis to a reference (ground truth) analysis.

    ### REFERENCE ANALYSIS (Ground Truth)
    {gt_analysis}

    ### GENERATED ANALYSIS
    {generated_analysis}

    ### EVALUATION RUBRIC (Rate 1-5 for each)

    **FACTUAL ACCURACY (1-5)**
    Do the key numerical values and facts in the generated analysis match those in the reference?
    Ignore differences in wording or style — only check whether the numbers and conclusions are correct.
    [1=Major errors or missing key numbers, 3=Mostly correct with minor deviations, 5=All key facts accurate]

    **COVERAGE (1-5)**
    Does the generated analysis cover the main points and conclusions present in the reference?
    [1=Missing most key points, 3=Main points covered, 5=All key points addressed]

    Respond ONLY with valid JSON in this exact format:
    {{
      "factual_accuracy": <1-5>,
      "coverage": <1-5>,
      "reasoning": "<brief explanation>"
    }}"""

    ANALYSIS_JUDGE_PROMPT_NO_GT = """You are an expert evaluator assessing a data analysis response.
    For the evaluation is important you consider the information that was available for the analysis, if the SQL result is wrong or has missing data, this problem shouldn't affect the analysis score.

    ### CONTEXT
    USER QUESTION: {prompt}
    SQL QUERY: {sql_query}
    SQL RESULTS:
    {data}

    ### ANALYSIS TO EVALUATE
    {analysis}

    ### EVALUATION RUBRIC (Rate 1-5 for each)

    **CORRECTNESS (1-5)**
    Does the analysis accurately interpret the SQL results? Are numerical values correct?
    [1=Wrong, 3=Mostly correct, 5=Perfect]

    **COMPLETENESS (1-5)**
    Does it fully address all parts of the user's question using available data?
    [1=Incomplete, 3=Main points covered, 5=Comprehensive]

    **FAITHFULNESS (1-5)**
    Does it only use information from SQL results? No hallucinated facts?
    [1=Major hallucinations, 3=Minor issues, 5=Fully grounded]

    ### OUTPUT
    Return ONLY valid JSON:
    {{
      "correctness": {{"score": <1-5>, "reasoning": "<brief>", "issues": []}},
      "completeness": {{"score": <1-5>, "reasoning": "<brief>", "missing": []}},
      "faithfulness": {{"score": <1-5>, "reasoning": "<brief>", "hallucinations": []}}
    }}"""

    def _batch_eval(self, states: list[State]) -> bool:
        return False

    def _eval(self, state: State, judge_provider: str, judge_model: str):
        last_analyzer_answer: Answer = state.get_last_answer("Analyzer")
        analysis = last_analyzer_answer.agent_output["analysis"]
        if not analysis:
            raise ValueError(
                f"The {State.__name__} did not contain a {'Analyzer'.value} {Answer.__name__}"
            )

        llm = get_llm(provider=judge_provider, model=judge_model)

        prompt = state.prompt
        last_retriever_answer: Answer = state.get_last_answer("Retriever")
        last_analyzer_answer: Answer = state.get_last_answer("Analyzer")
        sql_query: str = last_retriever_answer.agent_output["sql_query"]
        data: str = last_retriever_answer.agent_output["data_str"]
        analysis: str = last_analyzer_answer.agent_output["analysis"]

        truncated_data = data[:2000] if len(data) > 2000 else data

        formatted_prompt = AnalyzerEvaluator.ANALYSIS_JUDGE_PROMPT_NO_GT.format(
            prompt=prompt, sql_query=sql_query, data=truncated_data, analysis=analysis
        )

        response = llm.invoke(formatted_prompt)

        evaluation = fill_json_schema(response.extract_json(), _NO_GT_SCHEMA)
        score = compute_weighted_score(evaluation, _NO_GT_WEIGHTS)
        last_analyzer_answer.evaluation = Evaluation(score=score)

    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        analysis = answer.agent_output["analysis"]
        if not analysis:
            answer.evaluation = Evaluation(score=0)
            return

        llm = get_llm(provider=judge_provider, model=judge_model)
        gt_analysis = gt_data["analysis"]
        generated_analysis = answer.agent_output["analysis"]

        formatted_prompt = AnalyzerEvaluator.ANALYZE_JUDGE_PROMPT_GT.format(
            gt_analysis=gt_analysis,
            generated_analysis=generated_analysis,
        )
        response = llm.invoke(formatted_prompt)

        evaluation = fill_json_schema(
            _normalize_flat_judgement(response.extract_json()),
            _GT_SCHEMA,
        )
        score = compute_weighted_score(evaluation, _GT_WEIGHTS)
        answer.gt_evaluation = Evaluation(score=score)
        logger.debug(f"Evaluation successful : score={score}")

    def extract_gt_from_answer(self, answer: Answer) -> dict:
        if "analysis" in answer.agent_output:
            return {"analysis": answer.agent_output["analysis"]}
        return {}


__all__ = ["AnalyzerEvaluator"]
