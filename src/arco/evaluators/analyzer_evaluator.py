import json
from typing import Optional, Dict

from langchain_core.language_models import BaseChatModel

from arco.core import AgentType, State, Answer, Evaluation, Evaluator
from arco.core.llm_tools import get_llm


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

    @staticmethod
    def _parse_judge_json(raw_text: str) -> Dict:
        """Parse judge JSON response with robust error handling."""
        try:
            # Clean Markdown and find JSON
            content = raw_text.strip().replace("``````", "").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

            start = content.find("{")
            end = content.rfind("}")

            if start != -1 and end != -1:
                parsed = json.loads(content[start:end + 1])

                # Ensure all criteria exist
                for criterion in ["correctness", "completeness", "faithfulness"]:
                    if criterion not in parsed:
                        parsed[criterion] = {"score": 0, "reasoning": "Missing", "issues": []}

                return parsed
        except Exception as e:
            print(f"JSON parse error: {e}")

        # Fallback
        return {
            "correctness": {"score": 0, "reasoning": "Parse failed", "issues": []},
            "completeness": {"score": 0, "reasoning": "Parse failed", "missing": []},
            "faithfulness": {"score": 0, "reasoning": "Parse failed", "hallucinations": []}
        }

    @staticmethod
    def judge(state: State, llm: BaseChatModel):
        """Evaluate data analysis quality using LLM-as-a-Judge."""
        prompt = state.prompt
        last_retriever_answer: Answer = state.get_last_answer(AgentType.RETRIEVER)
        last_analyzer_answer: Answer = state.get_last_answer(AgentType.ANALYZER)
        sql_query: str = last_retriever_answer.agent_output['sql_query']
        data: str = last_retriever_answer.agent_output['data_str']
        analysis: str = last_analyzer_answer.agent_output['analysis']

        # Truncate data if too long
        truncated_data = data[:2000] if len(data) > 2000 else data

        # Get judgment
        formatted_prompt = AnalyzerEvaluator.ANALYSIS_JUDGE_PROMPT_NO_GT.format(
            prompt=prompt,
            sql_query=sql_query,
            data=truncated_data,
            analysis=analysis
        )

        response = llm.invoke(formatted_prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        evaluation = AnalyzerEvaluator._parse_judge_json(raw_content)

        # Compute overall score (average of 3 criteria)
        scores = [
            evaluation.get("correctness", {}).get("score", 0),
            evaluation.get("completeness", {}).get("score", 0),
            evaluation.get("faithfulness", {}).get("score", 0)
        ]
        score = sum(scores) / 3.0
        last_analyzer_answer.evaluation = Evaluation(score=(score - 1) / 4.0)
        return

    @staticmethod
    def judge_from_ground_truth(answer: Answer, llm: BaseChatModel, gt_analysis: Optional[str] = None) -> Evaluation:
        """Evaluate generated analysis against a ground truth reference using LLM-as-judge."""

        generated_analysis = answer.agent_output['analysis']

        formatted_prompt = AnalyzerEvaluator.ANALYZE_JUDGE_PROMPT_GT.format(
            gt_analysis=gt_analysis,
            generated_analysis=generated_analysis,
        )
        response = llm.invoke(formatted_prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Parse JSON response
        import re as _re
        json_match = _re.search(r'\{.*}', raw, _re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in judge response: {raw[:200]}")
        try:
            evaluation = json.loads(json_match.group())
        except Exception as _:
            return Evaluation(score=1)

        factual = float(evaluation.get("factual_accuracy", 1))
        coverage = float(evaluation.get("coverage", 1))
        score = ((factual + coverage) / 2 - 1) / 4  # normalize [1,5] → [0,1]
        return Evaluation(score=round(score, 4))

    def _eval(self, state: State, judge_provider: str, judge_model: str):
        last_analyzer_answer: Answer = state.get_last_answer(AgentType.ANALYZER)
        analysis = last_analyzer_answer.agent_output['analysis']
        if not analysis:
            raise ValueError(f"The {State.__name__} did not contain a {AgentType.ANALYZER.value} {Answer.__name__}")

        llm = get_llm(provider=judge_provider, model=judge_model)
        AnalyzerEvaluator.judge(state, llm)

    def _gt_eval(self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str):
        analysis = answer.agent_output['analysis']
        if not analysis:
            answer.evaluation = Evaluation(score=0)
            return

        llm = get_llm(provider=judge_provider, model=judge_model)
        evaluation = AnalyzerEvaluator.judge_from_ground_truth(
            answer=answer,
            llm=llm,
            gt_analysis=gt_data['analysis']
        )

        answer.gt_evaluation = evaluation
        return
