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
    cpp_csv_eval: bool = False,
    evaluator_exe: Optional[str] = None,
    eval_keys: Optional[str] = None,
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
        cpp_csv_eval: Use C++ CSV evaluator
        evaluator_exe: Path to C++ evaluator executable
        eval_keys: Comma-separated key columns for comparison
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
        elif cpp_csv_eval:
            if evaluator_exe is None:
                print("Cannot use --cpp_csv_eval because --evaluator-exe is not available") #TODO: make into warning

            keys = [k.strip() for k in (eval_keys or "").split(",") if k.strip()] or None
            def cpp_wrapper(csv_path):
                try:
                    output = run_cpp_comparator(
                        actual_csv=csv_path,
                        evaluator_exe=evaluator_exe,
                        expected_csv=gt_csv_path,
                        keys=keys
                    )
                    iou_type_map = {"columns": "columns_iou", "rows": "rows_iou", "table": "iou"}
                    return output.get(iou_type_map.get(iou_type,"rows_iou"),0.0)
                except Exception as e:
                    print(f"[CSV Eval] C++ comparator failed: {e}")
                    return 0.0
            csv_eval_fn = cpp_wrapper

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

def run_cpp_comparator(
    *,
    evaluator_exe: str,
    actual_csv: str,
    expected_csv: str,
    keys: Optional[List[str]] = None,
    case_insensitive: bool = False,
    stream_debug: bool = False,
) -> Dict:
    args = [evaluator_exe, "--actual", actual_csv, "--expected", expected_csv]
    if keys:
        args += ["--key", ",".join(keys)]
    if case_insensitive:
        args += ["--case-insensitive"]

    # If stream_debug is True, inherit stderr so C++ debug (sent to stderr) prints to terminal.
    # Keep stdout captured to parse JSON report.
    if stream_debug:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=None, text=True)
    else:
        proc = subprocess.run(args, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    try:
        report = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        report = {"equal": False, "error": "Invalid JSON from comparator", "raw": stdout}
    report["exit_code"] = proc.returncode
    if proc.returncode not in (0, 1):
        # Non-comparison error, include stderr
        report.setdefault("error", proc.stderr.strip())
    return report

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
    judge_model: str = "gpt-oss:20b",
    ollama_url: str = "http://localhost:11434"
) -> Tuple[float, Dict]:
    """Evaluate data analysis quality using LLM-as-a-Judge.
    
    Args:
        prompt: Original user question
        sql_query: SQL query that was executed
        data: SQL results (ground truth)
        analysis: LLM's analysis text to evaluate
        judge_model: Ollama model name for judging (default: llama3.1:70b)
        ollama_url: Ollama server URL
    
    Returns:
        float: Overall score (0,1) = average of correctness, completeness, faithfulness and the detailed_evaluation of the judge
    """
    from langchain_ollama import ChatOllama
    
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
