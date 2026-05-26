import json
import math
import os
import re
import subprocess
import tempfile
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco.core import AgentType, State, Answer, Evaluation, Evaluator
from arco.llm_tools import get_llm

if TYPE_CHECKING:
    from arco.core import AgentConfig


def _tokenize_for_bleu(text: str) -> List[str]:
    """Simple, dependency-free tokenization (words + numbers) for BLEU."""
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", (text or "").lower())


def bleu_score(hypothesis: str, reference: str, *, max_n: int = 4, smooth: bool = True) -> float:
    """Compute a simple BLEU score (0..1) with optional add-one smoothing.

    Intended for quick evaluation of generated analysis text; not a full SacreBLEU replacement.
    """

    ref_tokens = _tokenize_for_bleu(reference)
    hyp_tokens = _tokenize_for_bleu(hypothesis)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    def ngrams(tokens: List[str], n_grams: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i: i + n_grams]) for i in range(0, len(tokens) - n_grams + 1)]

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_ngrams = ngrams(hyp_tokens, n)
        ref_ngrams = ngrams(ref_tokens, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        hyp_counts: Dict[Tuple[str, ...], int] = {}
        ref_counts: Dict[Tuple[str, ...], int] = {}
        for g in hyp_ngrams:
            hyp_counts[g] = hyp_counts.get(g, 0) + 1
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1

        match = 0
        total = 0
        for g, c in hyp_counts.items():
            total += c
            match += min(c, ref_counts.get(g, 0))
        precisions.append((match + 1.0) / (total + 1.0) if smooth else (match / total if total else 0.0))

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / max(hyp_len, 1)))

    if any(p <= 0.0 for p in precisions):
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / float(max_n)
    return float(bp * math.exp(log_mean))


def spice_score_java(
        hypothesis: str,
        reference: str,
        *,
        spice_jar: str,
        java_bin: str = "java",
        timeout_seconds: int = 120,
) -> float:
    """Compute SPICE score (0..1) by calling the official Java SPICE jar.

    This uses the common COCO-caption SPICE JSON format:
      [{"image_id": 0, "test": "<candidate>", "refs": ["<ref1>", "<ref2>", ...]}]

    Args:
        reference: Ground-truth/reference text.
        hypothesis: Generated text to evaluate.
        spice_jar: Path to SPICE jar (e.g., spice-1.0.jar).
        java_bin: Java executable to use.
        timeout_seconds: Kill the Java process if it exceeds this time.

    Returns:
        SPICE F-score in [0,1].
    """
    if not spice_jar:
        raise ValueError("spice_jar is required")
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    # Use absolute paths to avoid cwd-related issues inside the Java tool
    spice_jar_abs = os.path.abspath(spice_jar)
    jar_dir = os.path.dirname(spice_jar_abs)

    payload = [
        {
            "image_id": 0,
            "test": hypothesis,
            "refs": [reference],
        }
    ]

    with tempfile.TemporaryDirectory() as td:
        in_json = os.path.abspath(os.path.join(td, "spice_in.json"))
        out_json = os.path.abspath(os.path.join(td, "spice_out.json"))
        with open(in_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        # Simple command: java -Xmx8G -jar spice-*.jar input.json
        cmd = [
            java_bin,
            "-Xmx8G",  # Add memory limit like your working command
            "-jar",
            spice_jar_abs,
            in_json,
            "-out",
            out_json,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                cwd=jar_dir,  # Run from jar directory
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"SPICE timed out after {timeout_seconds}s") from e
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise RuntimeError(f"SPICE failed: {stderr}") from e

        if not os.path.exists(out_json):
            raise RuntimeError("SPICE did not produce an output file")

        with open(out_json, "r", encoding="utf-8") as f:
            out = json.load(f)

        # Expected (COCO-caption): list with one element; element has `scores` -> `All` -> `f`
        item = out[0] if isinstance(out, list) else out
        scores = item.get("scores") or {}
        all_scores = scores.get("All") or scores.get("all") or {}
        f1 = all_scores.get("f") or all_scores.get("f1")
        return float(f1) if f1 is not None else 0.0


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

    def __init__(self, agent_config: AgentConfig):
        self.provider = agent_config.provider
        self.judge_model = agent_config.model
        self.ollama_url = agent_config.ollama_url
        self.metric = agent_config.gt_metric
        self.gt_text = agent_config.gt_text

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
        sql_query: str = last_retriever_answer.sql_query
        data: str = last_retriever_answer.data_str
        analysis: str = last_analyzer_answer.analysis

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
    def judge_from_ground_truth(state: State, llm: BaseChatModel, gt_analysis: Optional[str] = None) -> Evaluation:
        """Evaluate generated analysis against a ground truth reference using LLM-as-judge.

        Unlike judge_analysis() which scores against SQL data, this function compares
        the generated text directly to a reference (GT) text, checking whether key
        numerical facts and conclusions are captured correctly — regardless of phrasing."""

        last_analyzer_answer: Answer = state.get_last_answer(AgentType.ANALYZER)
        generated_analysis = last_analyzer_answer.analysis

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
        evaluation = json.loads(json_match.group())

        factual = float(evaluation.get("factual_accuracy", 1))
        coverage = float(evaluation.get("coverage", 1))
        score = ((factual + coverage) / 2 - 1) / 4  # normalize [1,5] → [0,1]
        return Evaluation(score=round(score, 4))

    def _eval(self, state: State):
        last_analyzer_answer: Answer = state.get_last_answer(AgentType.ANALYZER)
        analysis = last_analyzer_answer.analysis
        if not analysis:
            raise ValueError(f"The {State.__name__} did not contain a {AgentType.ANALYZER.value} {Answer.__name__}")

        llm = get_llm(provider=self.provider, model=self.judge_model, ollama_url=self.ollama_url)
        AnalyzerEvaluator.judge(state, llm)

    def _gt_eval(self, state: State):
        last_analyzer_answer: Answer = state.get_last_answer(AgentType.ANALYZER)
        analysis = last_analyzer_answer.analysis
        if not analysis:
            raise ValueError(f"The {State.__name__} did not contain a {AgentType.ANALYZER.value} {Answer.__name__}")

        if self.metric == "spice":
            score = spice_score_java(analysis, self.gt_text, spice_jar=None)
            evaluation = Evaluation(score=score)
        elif self.metric == "judge_gt":
            llm = get_llm(provider=self.provider, model=self.judge_model, ollama_url=self.ollama_url)
            evaluation = AnalyzerEvaluator.judge_from_ground_truth(
                state=state,
                llm=llm,
                gt_analysis=self.gt_text,
            )
        else:
            score = bleu_score(analysis, self.gt_text)
            evaluation = Evaluation(score=score)

        last_analyzer_answer.evaluation = evaluation
        return
