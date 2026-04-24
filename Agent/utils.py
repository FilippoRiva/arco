import pandas as pd
import sys
import os
import json
import numpy as np
import csv
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter
import math
import re
import subprocess
import tempfile
from functools import partial

def text_to_csv(text: str) -> List[List[str]]:
    """Convert text table to CSV rows.

    Handles both space-separated and pipe-separated formats.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return []

    rows = []
    for line in lines:
        # Try splitting by multiple spaces first
        if '  ' in line:
            parts = [p.strip() for p in line.split() if p.strip()]
        # Try pipe separator
        elif '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        # Fallback to comma
        else:
            parts = [p.strip() for p in line.split(',') if p.strip()]

        if parts:
            rows.append(parts)

    return rows

def text_to_dataframe(text: str) -> Optional[pd.DataFrame]:
    """Convert text table (from DataFrame.to_string()) back to a pandas DataFrame.

    This function handles the output format from DuckDB query results that have been
    converted to string using df.to_string(). It parses the column-aligned text format.

    Args:
        text: Text table string (space-separated columns with headers).

    Returns:
        pandas DataFrame or None if parsing fails.

    Example input format:
            date  sales  region
        0  2021-11-01    100  North
        1  2021-11-02    150  South
    """
    if not text or not text.strip():
        return None

    try:
        rows = text_to_csv(text)
        if not rows:
            return None

        # Detect index by comparing row lengths
        # If data rows have one more column than header row, it's likely the index
        has_index = False
        if len(rows) > 1:
            # Check if data rows have more columns than header
            if len(rows[1]) > len(rows[0]):
                has_index = True

        if has_index and len(rows) > 0:
            # Header row doesn't have index, use all columns
            # Data rows have index as first element, skip it
            headers = rows[0]
            data_rows = [row[1:] for row in rows[1:] if len(row) > 1]
        else:
            # No index column, first row is headers
            headers = rows[0]
            data_rows = rows[1:]

        if not headers or not data_rows:
            return None

        # Create DataFrame and infer types
        df = pd.DataFrame(data_rows, columns=headers)

        # Try to convert columns to appropriate types
        for col in df.columns:
            try:
                # Try numeric conversion
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Try datetime conversion
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    # Keep as string
                    pass

        return df

    except Exception as e:
        print(f"Error converting text to DataFrame: {e}")
        return None

def save_csv(rows: List[List[str]], filepath: str):
    """Save rows to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def get_evaluation_functions(
    *,
    lookup_only: bool = False,
    # CSV evaluation options
    gt_csv_path: Optional[str] = None,
    py_csv_eval: bool = False,
    iou_type: str = "rows",
    # Text evaluation options
    gt_text_path: Optional[str] = None,
    bleu_text_eval: bool = False,
    bleu_nltk: bool = False,
    spice_text_eval: bool = False,
    spice_jar: Optional[str] = None,
    spice_java_bin: str = "java",
    llm_text_eval: bool = False,
    llm_judge_model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    # Visualization evaluation options
    vis_eval: bool = False,
    gt_vis_config: Optional[Dict] = None,
    gt_vis_code: Optional[str] = None,
    vis_goal: Optional[str] = None,
    explicit_requirements: Optional[Dict] = None,
    vis_judge_model: str = "gpt-5.1",
    vis_provider: str = "openai",
    openai_api_key: Optional[str] = None,
) -> Tuple[Optional[callable], Optional[callable], Optional[callable]]:
    """Get evaluation functions based on command-line arguments.

    Args:
        lookup_only: If True, only CSV evaluation is relevant (no text analysis)
        py_csv_eval: Use Python CSV evaluator
        spice_text_eval: Use SPICE for text evaluation
        bleu_text_eval: Use BLEU for text evaluation
        llm_text_eval: Use LLM for text evaluation
        bleu_impl: BLEU implementation ("simple" or "nltk")
        spice_jar: Path to SPICE jar file
        spice_java_bin: Java executable for SPICE
        vis_eval: Enable visualization evaluation
        gt_vis_config: Ground truth chart configuration dict
        gt_vis_code: Ground truth matplotlib code string
        vis_goal: Visualization goal string
        explicit_requirements: Dict of explicit user requirements (color, formatting, etc.)
        vis_judge_model: Model for visualization judge (default: gpt-5.1)
        vis_provider: Provider for vis judge ("openai" or "ollama")
        openai_api_key: OpenAI API key (uses env var if not provided)

    Returns:
        Tuple of (csv_eval_fn, text_eval_fn, vis_eval_fn), any can be None
    """
    csv_eval_fn = None
    text_eval_fn = None
    vis_eval_fn = None

    # CSV Evaluation
    if gt_csv_path:
        if py_csv_eval:
            iou_type_map = {"columns": 0, "rows": 1, "table": 2}
            iou_index = iou_type_map.get(iou_type, 1)  # Default to rows (1)
            csv_eval_fn = lambda csv_path: compare_csv(csv_path, gt_csv_path)[iou_index]

    # Load ground truth if provided
    if gt_text_path:
        try:
            with open(gt_text_path, 'r', encoding='utf-8') as f:
                gt_text = f.read()
        except Exception as e:
            print(f"Failed to read expected analysis file: {str(e)}")
            gt_text = None

    if not lookup_only:
        if spice_text_eval and gt_text_path and gt_text:
            try:
                check_spice_jar_runnable(spice_jar=spice_jar, java_bin=spice_java_bin)
            except Exception as e:
                print(json.dumps({"error": f"SPICE precheck failed: {str(e)}"}, indent=2)) #TODO make into warning

            text_eval_fn = partial(spice_score_java, reference=gt_text, spice_jar=spice_jar, java_bin=spice_java_bin)

        elif bleu_text_eval and gt_text_path and gt_text:
            if bleu_nltk:
                text_eval_fn = partial(bleu_score_nltk,reference=gt_text, max_n=4, smooth=True)
            else:  # simple
                text_eval_fn = partial(bleu_score,reference=gt_text, max_n=4, smooth=True)

        elif llm_text_eval and llm_judge_model:
            def text_eval_llm(generated_text: str, prompt:str, sql_query:str, data:str) -> float:
                score, _ = judge_analysis(
                        prompt=prompt,
                        sql_query=sql_query,
                        data=data,
                        analysis=generated_text,
                        judge_model=llm_judge_model,
                        ollama_url=ollama_url
                    )
                return score
            text_eval_fn = text_eval_llm

        # Visualization Evaluation
        if vis_eval and gt_vis_config and gt_vis_code:
            def vis_eval_wrapper(chart_config: Dict, chart_code: str) -> float:
                score, _ = judge_visualization(
                    visualization_goal=vis_goal or "",
                    generated_config=chart_config,
                    generated_code=chart_code,
                    gt_config=gt_vis_config,
                    gt_code=gt_vis_code,
                    explicit_requirements=explicit_requirements,
                    judge_model=vis_judge_model,
                    provider=vis_provider,
                    openai_api_key=openai_api_key,
                    ollama_url=ollama_url or "http://localhost:11434"
                )
                return score
            vis_eval_fn = vis_eval_wrapper

    return csv_eval_fn, text_eval_fn, vis_eval_fn

def compare_csv(csv1_path, csv2_path):
    """
    Calculate IoU using multisets for proper duplicate handling.
    Column-order independent row comparison.
    """
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
    except Exception as e:
        print(f"Error while loading csvs for evaluation: {e}") 
        return 0. , 0. , 0.
    
    # Normalize column names to lowercase for case-insensitive comparison
    # (SQL aliases are case-insensitive; e.g. "total_revenue" == "Total_Revenue")
    df1.columns = df1.columns.str.lower()
    df2.columns = df2.columns.str.lower()

    # 1. Column names IoU
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    columns_names_iou = len(cols1 & cols2) / len(cols1 | cols2) if cols1 | cols2 else 0.0

    # 2. Overall data IoU
    data_counter1 = Counter(df1.values.flatten())
    data_counter2 = Counter(df2.values.flatten())

    intersection = data_counter1 & data_counter2
    union = data_counter1 | data_counter2
    data_iou = sum(intersection.values()) / sum(union.values()) if union else 0.0

    # 3. Row IoU
    cols_intersection = list(cols1 & cols2)
    if cols_intersection:
        sorted_cols = sorted(cols_intersection)  # Sort for consistency

        rows1 = [tuple(row) for row in df1[sorted_cols].values]
        rows2 = [tuple(row) for row in df2[sorted_cols].values]
        
        rows_counter1 = Counter(rows1)
        rows_counter2 = Counter(rows2)
        
        intersection = rows_counter1 & rows_counter2
        union = rows_counter1 | rows_counter2
        rows_iou = sum(intersection.values()) / sum(union.values()) if union else 0.0
        final_rows_iou = columns_names_iou * rows_iou
    else:
        final_rows_iou = 0.0
    
    return columns_names_iou, final_rows_iou, data_iou

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

    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]

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

def bleu_score_nltk(hypothesis: str, reference: str, *, max_n: int = 4, smooth: bool = True) -> float:
    """Compute BLEU (0..1) using NLTK's `sentence_bleu`.

    Requires:
        `pip install nltk`
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu  # type: ignore
        from nltk.translate.bleu_score import SmoothingFunction  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("NLTK is not installed. Install it with `pip install nltk`.") from e

    ref_tokens = _tokenize_for_bleu(reference)
    hyp_tokens = _tokenize_for_bleu(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    n = int(max(1, min(4, max_n)))
    if n == 1:
        weights = (1.0, 0.0, 0.0, 0.0)
    elif n == 2:
        weights = (0.5, 0.5, 0.0, 0.0)
    elif n == 3:
        weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)

    smoothing = SmoothingFunction().method1 if smooth else None
    score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothing)
    # NLTK returns a float in [0,1]
    return float(score)

def check_spice_jar_runnable(
    *,
    spice_jar: str,
    java_bin: str = "java",
    timeout_seconds: int = 10,
) -> None:
    """Fail-fast validation that the SPICE jar path exists and Java can execute it.

    This prevents spending time running the agent only to later fail with
    "Unable to access jarfile ...".
    """
    if not spice_jar:
        raise ValueError("spice_jar is required")

    jar_abs = os.path.abspath(spice_jar)
    if not os.path.exists(jar_abs):
        raise FileNotFoundError(f"SPICE jar not found: {jar_abs}")

    # If this is the common SPICE-1.0 bundle, ensure Stanford CoreNLP jars are present in lib/.
    jar_dir = os.path.dirname(jar_abs)
    lib_dir = os.path.join(jar_dir, "lib")
    if os.path.isdir(lib_dir):
        has_corenlp_code = any(
            fn.startswith("stanford-corenlp-") and fn.endswith(".jar") and "models" not in fn
            for fn in os.listdir(lib_dir)
        )
        has_corenlp_models = any(
            fn.startswith("stanford-corenlp-") and fn.endswith(".jar") and "models" in fn
            for fn in os.listdir(lib_dir)
        )
        if not (has_corenlp_code and has_corenlp_models):
            raise RuntimeError(
                "SPICE requires Stanford CoreNLP jars in the SPICE lib/ folder. "
                f"Missing in: {lib_dir}. "
                "The SPICE-1.0 bundle includes a script `get_stanford_models.sh` (Linux/macOS); "
                "on Windows, download CoreNLP 3.6.0 jars and place them into lib/ "
                "(both the code jar and the models jar)."
            )

        # On Windows, SPICE uses LMDB JNI. The bundle provides a win64 JNI jar; if Java is 32-bit,
        # it will fail at runtime with UnsatisfiedLinkError (lmdbjni32...).
        has_lmdb_win64 = any(fn.startswith("lmdbjni-win64-") and fn.endswith(".jar") for fn in os.listdir(lib_dir))
        if os.name == "nt" and has_lmdb_win64:
            try:
                ver = subprocess.run([java_bin, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds)
                ver_text = (ver.stderr or "") + "\n" + (ver.stdout or "")
                if "64-Bit" not in ver_text and "64-bit" not in ver_text:
                    raise RuntimeError(
                        "Your Java appears to be 32-bit, but SPICE-1.0 on Windows requires 64-bit Java "
                        "(lmdbjni-win64). Install a 64-bit JDK/JRE and ensure it is on PATH."
                    )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Java not found ('{java_bin}'). Install Java and ensure it's on PATH, or pass --spice-java-bin."
                ) from e

    cmd = [java_bin, "-jar", jar_abs]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=os.path.dirname(jar_abs) or None,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Java not found ('{java_bin}'). Install Java and ensure it's on PATH, or pass --spice-java-bin."
        ) from e
    except subprocess.TimeoutExpired:
        # If it runs longer than timeout, we assume the jar starts (good enough for this check).
        return

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    combined = (stderr + "\n" + stdout).strip().lower()

    if "unable to access jarfile" in combined:
        raise RuntimeError(f"Java cannot access the jar: {jar_abs}")
    if "no main manifest attribute" in combined:
        raise RuntimeError(f"Jar is not runnable (no main manifest attribute): {jar_abs}")

    # Otherwise: even if return code is non-zero, many jars print usage/help and exit -> OK.

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
            result = subprocess.run(
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
        try:
            item = out[0] if isinstance(out, list) else out
            scores = item.get("scores") or {}
            all_scores = scores.get("All") or scores.get("all") or {}
            f1 = all_scores.get("f") or all_scores.get("f1")
            return float(f1) if f1 is not None else 0.0
        except Exception:
            return 0.0
        
def judge_analysis(
    prompt: str,
    sql_query: str,
    data: str,
    analysis: str,
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    ollama_url: str = "http://localhost:11434",
    openai_api_key: Optional[str] = None,
) -> Tuple[float, Dict]:
    """Evaluate data analysis quality using LLM-as-a-Judge.

    Args:
        prompt: Original user question
        sql_query: SQL query that was executed
        data: SQL results (ground truth)
        analysis: LLM's analysis text to evaluate
        judge_model: Model name for judging (default: gpt-4o-mini)
        provider: LLM provider - "openai" or "ollama" (default: openai)
        ollama_url: Ollama server URL (only used when provider="ollama")
        openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)

    Returns:
        float: Overall score (0,1) = average of correctness, completeness, faithfulness and the detailed_evaluation of the judge
    """
    JUDGE_PROMPT = """You are an expert evaluator assessing a data analysis response.
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

    try:
        # Create judge LLM
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
            judge_llm = ChatOpenAI(
                model=judge_model,
                temperature=0.2,
                api_key=api_key,
                max_tokens=1000
            )
        else:
            from langchain_ollama import ChatOllama
            judge_llm = ChatOllama(
                model=judge_model,
                temperature=0.2,
                base_url=ollama_url,
                max_tokens=1000
            )
        
        # Truncate data if too long
        truncated_data = data[:2000] if len(data) > 2000 else data
        
        # Get judgment
        formatted_prompt = JUDGE_PROMPT.format(
            prompt=prompt,
            sql_query=sql_query,
            data=truncated_data,
            analysis=analysis
        )
        
        response = judge_llm.invoke(formatted_prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)
        
        # Parse JSON
        evaluation = _parse_judge_json(raw_content)
        
        # Compute overall score (average of 3 criteria)
        scores = [
            evaluation.get("correctness", {}).get("score", 0),
            evaluation.get("completeness", {}).get("score", 0),
            evaluation.get("faithfulness", {}).get("score", 0)
        ]
        score = sum(scores) / 3.0
        overall_score = (score - 1) / 4.0
        
        evaluation["overall_score"] = overall_score
        return overall_score, evaluation
            
    except Exception as e:
        print(f"Judge evaluation error: {e}")
        return (0.0, {"error": str(e)})


def _parse_judge_json(raw_text: str) -> Dict:
    """Parse judge JSON response with robust error handling."""
    try:
        # Clean markdown and find JSON
        content = raw_text.strip().replace("``````", "").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        
        start = content.find("{")
        end = content.rfind("}")
        
        if start != -1 and end != -1:
            parsed = json.loads(content[start:end+1])
            
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


# -----------------------------
# Visualization Evaluation (LLM-as-a-Judge)
# -----------------------------

VIS_JUDGE_PROMPT = """You are an expert data visualization evaluator. Your task is to assess whether a generated visualization achieves the same analytical purpose as a reference visualization.

## VISUALIZATION GOAL
{visualization_goal}

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


def _parse_vis_judge_json(raw_text: str) -> Dict:
    """Parse visualization judge JSON response with robust error handling."""
    try:
        content = raw_text.strip().replace("```json", "").replace("```", "").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:
            parsed = json.loads(content[start:end+1])

            # Ensure all criteria exist
            for criterion in ["axis_correctness", "chart_type", "functional_equivalence", "explicit_requirements"]:
                if criterion not in parsed:
                    parsed[criterion] = {"score": 1, "reasoning": "Missing", "violations": []}

            return parsed
    except Exception as e:
        print(f"Vis JSON parse error: {e}")

    # Fallback
    return {
        "axis_correctness": {"score": 1, "reasoning": "Parse failed", "x_match": False, "y_match": False},
        "chart_type": {"score": 1, "reasoning": "Parse failed", "type_match": False},
        "functional_equivalence": {"score": 1, "reasoning": "Parse failed", "would_render": False},
        "explicit_requirements": {"score": 5, "reasoning": "Parse failed - default N/A", "violations": []}
    }


def _compute_visualization_score(evaluation: Dict) -> float:
    """Compute weighted normalized score from judge evaluation.

    Returns:
        Score between 0.0 and 1.0
    """
    weights = {
        "axis_correctness": 0.40,
        "chart_type": 0.30,
        "functional_equivalence": 0.20,
        "explicit_requirements": 0.10
    }

    total_score = 0.0
    for criterion, weight in weights.items():
        raw_score = evaluation.get(criterion, {}).get("score", 1)
        # Normalize from 1-5 scale to 0-1
        normalized = (raw_score - 1) / 4.0
        total_score += normalized * weight

    return total_score


def judge_visualization(
    visualization_goal: str,
    generated_config: Dict[str, str],
    generated_code: str,
    gt_config: Dict[str, str],
    gt_code: str,
    explicit_requirements: Optional[Dict[str, Any]] = None,
    judge_model: str = "gpt-5.1",
    provider: str = "openai",
    openai_api_key: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.2,
) -> Tuple[float, Dict]:
    """Evaluate visualization quality using LLM-as-a-Judge.

    Args:
        visualization_goal: Original user request describing the desired visualization
        generated_config: Agent's generated chart_config dictionary
        generated_code: Agent's generated matplotlib code string
        gt_config: Ground truth chart_config dictionary
        gt_code: Ground truth matplotlib code string
        explicit_requirements: Dict of user-specified requirements that must be checked
            Keys: "color", "title_format", "label_format", "grid", "markers"
            Values: specific requirement string or None (not required)
        judge_model: Model name for the judge LLM (default: "gpt-5.1")
        provider: LLM provider ("openai" or "ollama")
        openai_api_key: API key for OpenAI (uses env var OPENAI_API_KEY if not provided)
        ollama_url: Ollama server URL (only used if provider="ollama")
        temperature: Sampling temperature for judge (default: 0.2)

    Returns:
        Tuple of (score: float, evaluation_details: Dict)
        - score: Normalized score between 0.0 and 1.0
        - evaluation_details: Dict with per-criterion scores and reasoning
    """
    import os

    try:
        # Create judge LLM based on provider
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
            judge_llm = ChatOpenAI(
                model=judge_model,
                temperature=temperature,
                api_key=api_key,
                max_tokens=1000
            )
        else:  # ollama
            from langchain_ollama import ChatOllama
            judge_llm = ChatOllama(
                model=judge_model,
                temperature=temperature,
                base_url=ollama_url,
                max_tokens=1000
            )

        # Format explicit requirements for display
        if explicit_requirements:
            req_display = "\n".join([
                f"- {k}: {v}" if v is not None else f"- {k}: (not specified - ignore)"
                for k, v in explicit_requirements.items()
            ])
        else:
            req_display = "None specified - ignore all styling requirements"

        # Truncate code if too long
        max_code_len = 2000
        gen_code_truncated = generated_code[:max_code_len] if len(generated_code) > max_code_len else generated_code
        gt_code_truncated = gt_code[:max_code_len] if len(gt_code) > max_code_len else gt_code

        # Format the judge prompt
        formatted_prompt = VIS_JUDGE_PROMPT.format(
            visualization_goal=visualization_goal,
            gt_config=json.dumps(gt_config, indent=2),
            gt_code=gt_code_truncated,
            gen_config=json.dumps(generated_config, indent=2),
            gen_code=gen_code_truncated,
            explicit_requirements=req_display
        )

        # Get judgment
        response = judge_llm.invoke(formatted_prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON response
        evaluation = _parse_vis_judge_json(raw_content)

        # Compute overall score
        overall_score = _compute_visualization_score(evaluation)
        evaluation["overall_score"] = overall_score

        return overall_score, evaluation

    except Exception as e:
        print(f"Visualization judge error: {e}")
        return (0.0, {"error": str(e)})


# -----------------------------
# Evaluator Factory Functions
# -----------------------------
# These factory functions create evaluation functions compatible with the
# per-step middleware. They have signature: (result: Dict, state: Dict) -> float

def make_csv_evaluator_gt(
    ground_truth_csv_path: Optional[str] = None,
    ground_truth_csv_text: Optional[str] = None,
) -> callable:
    """Factory to create CSV evaluation function for per-step execution.

    Compares the agent's result DataFrame against a ground-truth CSV using
    compare_dataframes_iou, which handles:
    - Float tolerance (atol=1e-2) to absorb precision differences from SQL casts

    Args:
        ground_truth_csv_path: Path to the ground truth CSV file.
        ground_truth_csv_text: Raw CSV text to use as ground truth.
            Exactly one of ground_truth_csv_path or ground_truth_csv_text must be provided.

    Returns:
        Function with signature (result: Dict, state: Dict) -> float
    """
    if ground_truth_csv_path and ground_truth_csv_text:
        raise ValueError("Provide exactly one of ground_truth_csv_path or ground_truth_csv_text, not both")
    if ground_truth_csv_text:
        gt_df = pd.read_csv(pd.io.common.StringIO(ground_truth_csv_text))
    elif ground_truth_csv_path:
        gt_df = pd.read_csv(ground_truth_csv_path)
    else:
        raise ValueError("Provide either ground_truth_csv_path or ground_truth_csv_text")

    _store: Dict = {}

    def eval_fn(result: Dict, state: Dict) -> float:
        data_df = result.get("data_df")
        if data_df is None:
            data_text = result.get("data", "")
            if data_text:
                data_df = text_to_dataframe(data_text)

        if data_df is None:
            _store["reasoning"] = "Model returned no data (SQL error or empty result)."
            return 0.0

        result_df = data_df.copy()
        result_df.columns = [c.lower() for c in result_df.columns]

        gt_df_cmp = gt_df.copy()
        gt_df_cmp.columns = [c.lower() for c in gt_df_cmp.columns]

        try:
            score = compare_dataframes_iou(result_df, gt_df_cmp)
            if score < 1.0:
                n_gt = len(gt_df_cmp)
                n_model = len(result_df)
                gt_cols = set(gt_df_cmp.columns.tolist())
                model_cols = set(result_df.columns.tolist())
                if n_model == 0:
                    _store["reasoning"] = "Model returned empty result."
                elif gt_cols != model_cols:
                    missing = sorted(gt_cols - model_cols)
                    extra = sorted(model_cols - gt_cols)
                    parts = [f"GT {n_gt} rows, model {n_model} rows, IOU={score:.3f}."]
                    if missing:
                        parts.append(f"Missing cols: {missing}.")
                    if extra:
                        parts.append(f"Extra cols: {extra}.")
                    _store["reasoning"] = " ".join(parts)
                else:
                    try:
                        gt_r0 = dict(zip(gt_df_cmp.columns, gt_df_cmp.iloc[0].tolist()))
                        mod_r0 = dict(zip(result_df.columns, result_df.iloc[0].tolist())) if n_model > 0 else {}
                        _store["reasoning"] = (
                            f"GT {n_gt} rows, model {n_model} rows, IOU={score:.3f}. "
                            f"GT row[0]={gt_r0} | Model row[0]={mod_r0}"
                        )
                    except Exception:
                        _store["reasoning"] = (
                            f"GT {n_gt} rows, model {n_model} rows, IOU={score:.3f}. "
                            "Same columns but values differ."
                        )
            return score
        except Exception as e:
            _store["reasoning"] = f"Evaluation error: {e}"
            print(f"CSV evaluation error: {e}")
            return 0.0

    eval_fn._store = _store
    return eval_fn


def judge_analysis_gt(
    generated_analysis: str,
    gt_analysis: str,
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    ollama_url: str = "http://localhost:11434",
    openai_api_key: Optional[str] = None,
) -> Tuple[float, Dict]:
    """Evaluate generated analysis against a ground truth reference using LLM-as-judge.

    Unlike judge_analysis() which scores against SQL data, this function compares
    the generated text directly to a reference (GT) text, checking whether key
    numerical facts and conclusions are captured correctly — regardless of phrasing.

    Args:
        generated_analysis: The analysis text produced by the agent.
        gt_analysis: The reference (ground truth) analysis text.
        judge_model: LLM model for judging (default: gpt-4o-mini).
        provider: LLM provider — 'openai' or 'ollama'.
        ollama_url: Ollama server URL (only used when provider='ollama').
        openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).

    Returns:
        (score: float in [0, 1], evaluation: Dict with per-criterion scores and reasoning)
    """
    JUDGE_GT_PROMPT = """You are an expert evaluator comparing a generated data analysis to a reference (ground truth) analysis.

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

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            judge_llm = ChatOpenAI(model=judge_model, temperature=0.0, api_key=api_key)
        else:
            from langchain_ollama import ChatOllama
            judge_llm = ChatOllama(model=judge_model, temperature=0.0, base_url=ollama_url)

        formatted_prompt = JUDGE_GT_PROMPT.format(
            gt_analysis=gt_analysis,
            generated_analysis=generated_analysis,
        )
        response = judge_llm.invoke(formatted_prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Parse JSON response
        import re as _re
        json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in judge response: {raw[:200]}")
        evaluation = json.loads(json_match.group())

        factual = float(evaluation.get("factual_accuracy", 1))
        coverage = float(evaluation.get("coverage", 1))
        score = ((factual + coverage) / 2 - 1) / 4  # normalize [1,5] → [0,1]
        evaluation["overall_score"] = round(score, 4)
        return score, evaluation

    except Exception as e:
        print(f"GT analysis judge error: {e}")
        return 0.0, {"error": str(e)}


def make_text_evaluator_gt(
    ground_truth_text: str,
    metric: str = "judge_gt",
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    ollama_url: str = "http://localhost:11434",
    openai_api_key: Optional[str] = None,
) -> callable:
    """Factory to create a GT-based text evaluation function for per-step execution.

    Args:
        ground_truth_text: Reference analysis text to compare against (required).
        metric: Evaluation metric — 'bleu', 'spice', or 'judge_gt' (default).
            - 'bleu'     : n-gram overlap; fast but penalises valid paraphrases.
            - 'spice'    : semantic similarity via scene graph; requires Java.
            - 'judge_gt' : LLM judge that checks factual accuracy and coverage
                           against the GT text, ignoring surface-level phrasing.
        judge_model: LLM model for judge (default: gpt-4o-mini). Used only for 'judge_gt'.
        provider: LLM provider — 'openai' or 'ollama'. Used only for 'judge_gt'.
        ollama_url: Ollama server URL. Used only when provider='ollama'.
        openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).

    Returns:
        Function with signature (result: Dict, state: Dict) -> float
    """
    _store: Dict = {}

    def eval_fn(result: Dict, state: Dict) -> float:
        answers = result.get("answer", [])
        if not answers:
            return 0.0
        analysis_text = answers[0] if isinstance(answers, list) else str(answers)
        if not analysis_text:
            return 0.0

        try:
            if metric == "bleu":
                return bleu_score(analysis_text, ground_truth_text)
            elif metric == "spice":
                return spice_score_java(analysis_text, ground_truth_text)
            elif metric == "judge_gt":
                score, evaluation = judge_analysis_gt(
                    generated_analysis=analysis_text,
                    gt_analysis=ground_truth_text,
                    judge_model=judge_model,
                    provider=provider,
                    ollama_url=ollama_url,
                    openai_api_key=openai_api_key,
                )
                print(f"[analyzing_data GT judge] factual_accuracy={evaluation.get('factual_accuracy')} | coverage={evaluation.get('coverage')} | reasoning: {evaluation.get('reasoning', 'N/A')}")
                if score < 1.0:
                    _store["reasoning"] = (
                        f"factual_accuracy={evaluation.get('factual_accuracy')}, "
                        f"coverage={evaluation.get('coverage')}. "
                        f"{evaluation.get('reasoning', '')}"
                    )
                return score
            else:
                return bleu_score(analysis_text, ground_truth_text)

        except Exception as e:
            _store["reasoning"] = f"Evaluation error: {e}"
            print(f"Text GT evaluation error: {e}")
            return 0.0

    eval_fn._store = _store
    return eval_fn


def make_vis_evaluator_gt(
    ground_truth_config: Dict,
    ground_truth_code: str,
    explicit_requirements: Optional[Dict] = None,
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    openai_api_key: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
) -> callable:
    """Factory to create visualization evaluation function for per-step execution.

    Args:
        ground_truth_config: Expected chart configuration dict.
        ground_truth_code: Expected chart code string.
        explicit_requirements: Optional dict of explicit styling requirements.
        judge_model: LLM model for the judge (default: gpt-4o-mini).
        provider: LLM provider — 'openai' or 'ollama'.
        openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).
        ollama_url: Ollama server URL (only used when provider='ollama').

    Returns:
        Function with signature (result: Dict, state: Dict) -> float
        that extracts chart_config and code from result and evaluates.
    """
    _store: Dict = {}

    def eval_fn(result: Dict, state: Dict) -> float:
        chart_config = result.get("chart_config")
        answers = result.get("answer", [])

        if not chart_config:
            _store["reasoning"] = "Model produced no chart_config."
            return 0.0

        # Chart code is the last answer entry
        chart_code = answers[-1] if answers else None
        if not chart_code:
            _store["reasoning"] = "Model produced no chart code."
            return 0.0

        visualization_goal = state.get("visualization_goal", state.get("prompt", ""))

        try:
            score, evaluation = judge_visualization(
                visualization_goal=visualization_goal,
                generated_config=chart_config,
                generated_code=chart_code,
                gt_config=ground_truth_config,
                gt_code=ground_truth_code,
                explicit_requirements=explicit_requirements,
                judge_model=judge_model,
                provider=provider,
                openai_api_key=openai_api_key,
                ollama_url=ollama_url,
            )
            if score < 1.0:
                reasoning_parts = []
                for key, val in evaluation.items():
                    if key not in ("overall_score", "error") and val is not None:
                        reasoning_parts.append(f"{key}={val}")
                _store["reasoning"] = "; ".join(reasoning_parts) if reasoning_parts else str(evaluation)
            return score

        except Exception as e:
            _store["reasoning"] = f"Evaluation error: {e}"
            print(f"Visualization evaluation error: {e}")
            return 0.0

    eval_fn._store = _store
    return eval_fn


# =========================================================
# No-Ground-Truth Evaluator Factory Functions
# =========================================================
# These factory functions create evaluation functions that do NOT require
# ground truth data. They enable meaningful best-of-N selection during
# normal (non-evaluation) usage.


def compare_dataframes_iou(df1: pd.DataFrame, df2: pd.DataFrame, atol: float = 1e-2) -> float:
    """Compute row-level IoU between two DataFrames.

    Column selection strategy:
    - Exact same columns in the same order → compare all columns positionally.
    - Otherwise, compare only shared columns with exact column-name matches.
    - If there are no shared columns, return 0.0.

    Numeric values are compared with absolute tolerance ``atol``.

    Returns:
        rows_iou (float in [0, 1])
    """
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return 0.0

    # Normalize values (dates, floats, ints, strings) for consistent comparison
    df1 = normalize_dataframe_values(df1)
    df2 = normalize_dataframe_values(df2)

    if list(df1.columns) == list(df2.columns):
        v1 = df1.values
        v2 = df2.values
    else:
        # Compare only columns that already share the exact same names.
        shared = [col for col in df1.columns if col in df2.columns]
        if not shared:
            return 0.0
        v1 = df1[shared].values
        v2 = df2[shared].values

    def _row_matches(r1, r2):
        """Check if two row arrays match element-wise with float tolerance."""
        if len(r1) != len(r2):
            return False
        for a, b in zip(r1, r2):
            try:
                if abs(float(a) - float(b)) <= atol:
                    continue
            except (ValueError, TypeError):
                pass
            # Normalize: strip trailing midnight time from date strings
            sa = str(a).replace(" 00:00:00", "")
            sb = str(b).replace(" 00:00:00", "")
            if sa == sb:
                continue
            # Handle YYYY-MM vs YYYY-MM-DD: treat first-of-month as equivalent
            if sa[:7] == sb[:7] and (
                (len(sa) == 7 and len(sb) == 10 and sb.endswith("-01")) or
                (len(sb) == 7 and len(sa) == 10 and sa.endswith("-01"))
            ):
                continue
            return False
        return True

    # Greedy matching: for each row in v1, find an unmatched row in v2
    used = [False] * len(v2)
    matched = 0
    for row1 in v1:
        for j, row2 in enumerate(v2):
            if not used[j] and _row_matches(row1, row2):
                used[j] = True
                matched += 1
                break

    total = len(v1) + len(v2) - matched  # union count
    return matched / total if total > 0 else 0.0


def normalize_dataframe_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cell values in a DataFrame for consistent comparison.

    Per-column transformations based on auto-detected type:
    - Numeric integers: cast float-encoded ints (e.g., 33653.0 → 33653)
    - Numeric floats: round to 2 decimal places
    - Dates: format as YYYY-MM-DD string
    - Strings: strip leading/trailing whitespace
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    for col in df.columns:
        series = df[col]

        # Handle datetime dtype first (before numeric, since datetime64 is numeric-castable)
        if pd.api.types.is_datetime64_any_dtype(series):
            df[col] = series.dt.strftime("%Y-%m-%d")
            continue

        # Try numeric
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and not series.isna().all():
            # Check if all values are whole numbers → cast to int
            if (numeric == numeric.round(0)).all():
                df[col] = numeric.astype(int)
            else:
                df[col] = numeric.round(2)
            continue

        # Try datetime (for string columns like "2023-01-01")
        try:
            dt = pd.to_datetime(series, errors="coerce", format="mixed")
            if dt.notna().all() and not series.isna().all():
                df[col] = dt.dt.strftime("%Y-%m-%d")
                continue
        except Exception:
            pass

        # Fallback: string strip
        if series.dtype == object:
            df[col] = series.astype(str).str.strip()

    return df


COLUMN_STANDARDIZATION_PROMPT = """\
You are a data schema expert. Given N SQL queries against the same database that \
answer the same question, standardize their result column names and order.

## Database Schema
{schema_context}

## Candidates
{candidates_section}

## Rules
- For columns that come directly from schema tables, use the exact schema column name.
- For aggregated/computed columns (SUM, COUNT, AVG, etc.), pick the most descriptive \
name used by any candidate. Prefer lowercase_with_underscores.
- All candidates MUST map to the same canonical columns in the same order.
- Return ONLY valid JSON, no explanation or markdown fences.

## Output format
{{"canonical_columns": ["col1", "col2"], "mappings": [{{"original_col": "canonical_col", ...}}, ...]}}
"""


def standardize_candidate_columns(
    results: List[Dict],
    schema,
    llm,
    gt_columns: Optional[List[str]] = None,
) -> List[Dict]:
    """Use an LLM to standardize column names across best-of-n candidates.

    After best-of-n generates N SQL results, their DataFrames may have different
    column names and orders. This function asks the LLM to determine canonical
    column names and reorders/renames each candidate's DataFrame to match.

    Also applies normalize_dataframe_values to each DataFrame.

    Args:
        results: List of result dicts from best-of-n, each with 'data_df' and 'sql_query'.
        schema: DatabaseSchema instance for context.
        llm: LangChain LLM instance (should use temperature=0).

    Returns:
        Modified results list with standardized DataFrames. On error, returns
        results unchanged.
    """
    import json as _json

    # Collect candidate info
    candidates_info = []
    for i, r in enumerate(results):
        df = r.get("data_df")
        sql = r.get("sql_query", "N/A")
        cols = list(df.columns) if df is not None else []
        candidates_info.append({"idx": i, "sql": sql, "cols": cols, "df": df})

    # Skip if fewer than 2 candidates have DataFrames
    valid = [c for c in candidates_info if c["df"] is not None and len(c["cols"]) > 0]

    def _apply_gt_alignment(df: pd.DataFrame, canonical_cols: list) -> pd.DataFrame:
        """Rename and reorder df columns to match canonical_cols without LLM."""
        df = df.copy()
        current_cols = list(df.columns)
        if len(current_cols) == len(canonical_cols):
            # Case-insensitive rename
            ci_map = {c.lower(): c for c in current_cols}
            fixed = {ci_map[canon.lower()]: canon
                     for canon in canonical_cols
                     if ci_map.get(canon.lower()) and ci_map[canon.lower()] != canon}
            if fixed:
                df = df.rename(columns=fixed)
            # Positional rename as last resort
            if list(df.columns) != canonical_cols and len(df.columns) == len(canonical_cols):
                df.columns = canonical_cols
        # Reorder to canonical order if all columns present
        if set(canonical_cols).issubset(set(df.columns)):
            df = df[canonical_cols]
        return normalize_dataframe_values(df)

    if len(valid) < 2:
        # Special case: single candidate + gt_columns → apply GT column alignment without LLM
        if len(valid) == 1 and gt_columns:
            ci = valid[0]
            idx = ci["idx"]
            df = _apply_gt_alignment(results[idx]["data_df"], list(gt_columns))
            results[idx]["data_df"] = df
            results[idx]["data"] = df.to_csv(index=False)
            print(f"[standardize] Single-candidate GT alignment → columns: {list(df.columns)}")
        return results

    # When all candidates share the same columns AND gt_columns is provided but
    # column names don't already match GT → apply GT alignment directly to all
    # candidates without calling the LLM (no inter-candidate disagreement to resolve).
    col_lists = [tuple(ci["cols"]) for ci in valid]
    if len(set(col_lists)) == 1:
        if gt_columns is None:
            return results
        current_lower = [c.lower() for c in valid[0]["cols"]]
        if current_lower == [c.lower() for c in gt_columns]:
            return results
        # All candidates agree but names don't match GT → rename all without LLM
        canonical_cols = list(gt_columns)
        for ci in valid:
            idx = ci["idx"]
            df = _apply_gt_alignment(results[idx]["data_df"], canonical_cols)
            results[idx]["data_df"] = df
            results[idx]["data"] = df.to_csv(index=False)
        print(f"[standardize] Multi-candidate GT alignment (same cols) → columns: {canonical_cols}")
        return results

    # Build prompt
    schema_context = schema.get_full_schema_str() if schema else "No schema available"
    candidates_lines = []
    for ci in valid:
        candidates_lines.append(
            f"Candidate {ci['idx'] + 1}: SQL: {ci['sql']} | Columns: {ci['cols']}"
        )
    candidates_section = "\n".join(candidates_lines)

    if gt_columns:
        gt_hint = (
            f"\n## Required Output Column Names (Ground Truth)\n"
            f"The canonical_columns in your output MUST be exactly: {gt_columns} (in this order).\n"
            f"Rename each candidate column to its semantically matching entry in this list.\n"
            f"Do NOT use schema column names or candidate names — use only these GT names."
        )
        prompt = COLUMN_STANDARDIZATION_PROMPT.format(
            schema_context=schema_context,
            candidates_section=candidates_section,
        ) + gt_hint
    else:
        prompt = COLUMN_STANDARDIZATION_PROMPT.format(
            schema_context=schema_context,
            candidates_section=candidates_section,
        )

    # Call LLM
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    # Parse JSON — strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    mapping_data = _json.loads(raw)
    canonical_cols = mapping_data["canonical_columns"]
    mappings = mapping_data["mappings"]

    if len(mappings) != len(valid):
        print(f"[standardize] Warning: expected {len(valid)} mappings, got {len(mappings)}")
        return results

    # Apply mappings
    for ci, col_map in zip(valid, mappings):
        idx = ci["idx"]
        df = results[idx]["data_df"]
        if df is None:
            continue

        rename_map = {old: new for old, new in col_map.items() if old in df.columns}
        df = df.rename(columns=rename_map)

        # Reorder to canonical order (only if all canonical cols are present)
        if set(canonical_cols).issubset(set(df.columns)):
            df = df[canonical_cols]

        # Normalize values
        df = normalize_dataframe_values(df)

        # Update result
        results[idx]["data_df"] = df
        results[idx]["data"] = df.to_csv(index=False)

    print(f"[standardize] Standardized {len(valid)} candidates → columns: {canonical_cols}")
    return results


def make_csv_evaluator_no_gt() -> callable:
    """Factory to create a consensus-based CSV evaluator (no ground truth).

    Returns a batch_eval_fn with signature:
        (results: List[Dict], state: Dict) -> List[float]

    Each result's score is its average pairwise row-IoU to all other results.
    The most "agreed upon" DataFrame wins.
    """
    def batch_eval_fn(results: List[Dict], state: Dict) -> List[float]:
        # Extract DataFrames from results
        dfs = []
        for r in results:
            df = r.get("data_df")
            if df is None:
                data_text = r.get("data", "")
                if data_text:
                    df = text_to_dataframe(data_text)
            dfs.append(df)

        n = len(dfs)
        if n <= 1:
            return [1.0] * n

        # Compute pairwise IoU matrix
        scores = [0.0] * n
        for i in range(n):
            if dfs[i] is None:
                continue
            total = 0.0
            count = 0
            for j in range(n):
                if i == j or dfs[j] is None:
                    continue
                total += compare_dataframes_iou(dfs[i], dfs[j])
                count += 1
            scores[i] = total / count if count > 0 else 0.0

        return scores

    return batch_eval_fn


def make_text_evaluator_no_gt(
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    ollama_url: str = "http://localhost:11434",
    openai_api_key: Optional[str] = None,
) -> callable:
    """Factory to create a text evaluator without ground truth.

    Uses the existing judge_analysis() which scores analysis quality
    based on correctness, completeness, and faithfulness against the
    SQL data — no ground truth text needed.

    Returns:
        Function with signature (result: Dict, state: Dict) -> float
    """
    def eval_fn(result: Dict, state: Dict) -> float:
        answers = result.get("answer", [])
        if not answers:
            return 0.0

        analysis_text = answers[0] if isinstance(answers, list) else str(answers)
        if not analysis_text:
            return 0.0

        try:
            score, _ = judge_analysis(
                prompt=state.get("prompt", result.get("prompt", "")),
                sql_query=state.get("sql_query", result.get("sql_query", "")),
                data=state.get("data", result.get("data", "")),
                analysis=analysis_text,
                judge_model=judge_model,
                provider=provider,
                ollama_url=ollama_url,
                openai_api_key=openai_api_key,
            )
            return score
        except Exception as e:
            print(f"No-GT text evaluation error: {e}")
            return 0.0

    return eval_fn


# --- No-GT Visualization Judge ---

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


def _parse_vis_no_gt_judge_json(raw_text: str) -> Dict:
    """Parse no-GT visualization judge JSON response."""
    try:
        content = raw_text.strip().replace("```json", "").replace("```", "").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:
            parsed = json.loads(content[start:end+1])
            for criterion in ["data_suitability", "axis_mapping", "code_quality", "goal_alignment"]:
                if criterion not in parsed:
                    parsed[criterion] = {"score": 1, "reasoning": "Missing"}
            return parsed
    except Exception as e:
        print(f"Vis no-GT JSON parse error: {e}")

    return {
        "data_suitability": {"score": 1, "reasoning": "Parse failed"},
        "axis_mapping": {"score": 1, "reasoning": "Parse failed", "columns_exist": False},
        "code_quality": {"score": 1, "reasoning": "Parse failed", "would_render": False},
        "goal_alignment": {"score": 1, "reasoning": "Parse failed"},
    }


def _compute_vis_no_gt_score(evaluation: Dict) -> float:
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


def judge_visualization_no_gt(
    visualization_goal: str,
    generated_config: Dict[str, str],
    generated_code: str,
    data_columns: List[str],
    data_sample: str,
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    openai_api_key: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.2,
) -> Tuple[float, Dict]:
    """Evaluate visualization quality without ground truth using LLM-as-a-Judge.

    Args:
        visualization_goal: User's visualization request.
        generated_config: Agent's chart_config dictionary.
        generated_code: Agent's matplotlib code string.
        data_columns: List of column names available in the data.
        data_sample: String representation of first few rows of data.
        judge_model: Model for the judge LLM.
        provider: "openai" or "ollama".
        openai_api_key: OpenAI API key (uses env var if not provided).
        ollama_url: Ollama server URL.
        temperature: Sampling temperature for judge.

    Returns:
        Tuple of (score: float 0-1, evaluation_details: Dict)
    """
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
            judge_llm = ChatOpenAI(
                model=judge_model, temperature=temperature,
                api_key=api_key, max_tokens=1000
            )
        else:
            from langchain_ollama import ChatOllama
            judge_llm = ChatOllama(
                model=judge_model, temperature=temperature,
                base_url=ollama_url, max_tokens=1000
            )

        max_code_len = 2000
        gen_code_truncated = generated_code[:max_code_len] if len(generated_code) > max_code_len else generated_code

        formatted_prompt = VIS_JUDGE_NO_GT_PROMPT.format(
            visualization_goal=visualization_goal,
            data_columns=", ".join(data_columns),
            data_sample=data_sample[:1500],
            gen_config=json.dumps(generated_config, indent=2),
            gen_code=gen_code_truncated,
        )

        response = judge_llm.invoke(formatted_prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)

        evaluation = _parse_vis_no_gt_judge_json(raw_content)
        overall_score = _compute_vis_no_gt_score(evaluation)
        evaluation["overall_score"] = overall_score

        return overall_score, evaluation

    except Exception as e:
        print(f"No-GT visualization judge error: {e}")
        return (0.0, {"error": str(e)})


def make_vis_evaluator_no_gt(
    judge_model: str = "gpt-4o-mini",
    provider: str = "openai",
    ollama_url: str = "http://localhost:11434",
    openai_api_key: Optional[str] = None,
) -> callable:
    """Factory to create a visualization evaluator without ground truth.

    Uses an LLM judge to score chart quality based on data suitability,
    axis mapping, code quality, and goal alignment.

    Returns:
        Function with signature (result: Dict, state: Dict) -> float
    """
    def eval_fn(result: Dict, state: Dict) -> float:
        chart_config = result.get("chart_config")
        answers = result.get("answer", [])

        if not chart_config:
            return 0.0

        chart_code = answers[-1] if answers else None
        if not chart_code:
            return 0.0

        # Get data columns and sample from result or state
        data_df = result.get("data_df", state.get("data_df"))
        if data_df is not None and hasattr(data_df, 'columns'):
            data_columns = list(data_df.columns)
            data_sample = data_df.head(5).to_string(index=False)
        else:
            data_text = result.get("data", state.get("data", ""))
            data_columns = []
            data_sample = data_text[:500] if data_text else ""

        vis_goal = state.get("visualization_goal", state.get("prompt", ""))

        try:
            score, evaluation = judge_visualization_no_gt(
                visualization_goal=vis_goal,
                generated_config=chart_config,
                generated_code=chart_code,
                data_columns=data_columns,
                data_sample=data_sample,
                judge_model=judge_model,
                provider=provider,
                openai_api_key=openai_api_key,
                ollama_url=ollama_url,
            )
            for criterion, detail in evaluation.items():
                if not isinstance(detail, dict):
                    continue
                raw = detail.get("score", "?")
                reasoning = detail.get("reasoning", "")
                print(f"  [vis eval] {criterion}: {raw}/5 — {reasoning}")
            return score
        except Exception as e:
            print(f"No-GT visualization evaluation error: {e}")
            return 0.0

    return eval_fn
