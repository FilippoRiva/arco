from io import StringIO

import pandas as pd

from arco.core import AgentType, Answer, Evaluation, Evaluator, State
from arco.data import normalize_dataframe_values


def compare_dataframes_iou(
    df1: pd.DataFrame, df2: DataFrame, atol: float = 1e-2
) -> float:
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
            except ValueError, TypeError:
                pass
            # Normalize: strip trailing midnight time from date strings
            sa = str(a).replace(" 00:00:00", "")
            sb = str(b).replace(" 00:00:00", "")
            if sa == sb:
                continue
            # Handle YYYY-MM vs YYYY-MM-DD: treat first-of-month as equivalent
            if sa[:7] == sb[:7] and (
                (len(sa) == 7 and len(sb) == 10 and sb.endswith("-01"))
                or (len(sb) == 7 and len(sa) == 10 and sa.endswith("-01"))
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


class RetrieverEvaluator(Evaluator):
    def _batch_eval(self, states: list[State]):
        """
        Each result's score is its average pairwise row-IoU to all other results.
        The most "agreed upon" DataFrame wins.
        """

        # Extract DataFrames from results
        # pyrefly: ignore [bad-assignment]
        answers: list[Answer] = [r.get_last_answer(AgentType.RETRIEVER) for r in states]
        if None in answers:
            raise ValueError(
                f"One {State.__name__} did not contain a {AgentType.RETRIEVER.value} {Answer.__name__}"
            )

        # Default when Best-of-1
        if len(answers) == 1:
            answers[0].evaluation = Evaluation(score=1.0)
            return True

        dfs = [a.agent_output["data_df"] for a in answers]

        # Compute pairwise IoU matrix
        for i in range(len(dfs)):
            if dfs[i] is None:
                continue
            total = 0.0
            count = 0
            for j in range(len(dfs)):
                if i == j or dfs[j] is None:
                    continue
                total += compare_dataframes_iou(dfs[i], dfs[j])  # pyrefly: ignore [bad-argument-type]
                count += 1
            answers[i].evaluation = Evaluation(
                score=total / count if count > 0 else 0.0
            )

        return True  # if success

    @staticmethod
    def _apply_gt_alignment(answer: Answer, canonic_cols: list):
        """Rename and reorder df columns to match canonical_cols without LLM."""
        if answer is None:
            raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)
        if answer.agent_output["data_df"] is None:
            raise AgentException(missing_dataframe_from_type=AgentType.RETRIEVER)
        df_to_align: pd.DataFrame = answer.agent_output["data_df"]
        current_cols = list(df_to_align.columns)
        if len(current_cols) == len(canonic_cols):
            # Case-insensitive rename
            ci_map = {c.lower(): c for c in current_cols}
            fixed = {
                ci_map[canon.lower()]: canon
                for canon in canonic_cols
                if ci_map.get(canon.lower()) and ci_map[canon.lower()] != canon
            }
            if fixed:
                df_to_align = df_to_align.rename(columns=fixed)
            # Positional rename as last resort
            if list(df_to_align.columns) != canonic_cols and len(
                df_to_align.columns
            ) == len(canonic_cols):
                df_to_align.columns = canonic_cols
        # Reorder to canonical order if all columns present
        if set(canonic_cols).issubset(set(df_to_align.columns)):
            # pyrefly: ignore [bad-assignment]
            df_to_align = df_to_align[canonic_cols]

        # Normalize
        normalized_df: pd.DataFrame = normalize_dataframe_values(df_to_align)
        # Assign normalized and aligned dataframe
        answer.agent_output["data_df"] = normalized_df
        answer.agent_output["data_str"] = normalized_df.to_csv(index=False)

    def _gt_eval(
        self, answer: Answer, gt_data: dict, judge_provider: str, judge_model: str
    ):
        """
        Compares the agent's result DataFrame against a ground-truth CSV using
        compare_dataframes_iou, which handles:
        - Float tolerance (atol=1e-2) to absorb precision differences from SQL casts
        """
        if not answer:
            raise ValueError(
                f"Tried to evaluate a {State.__name__} with no {AgentType.RETRIEVER.value} {Answer.__name__} with a {RetrieverEvaluator.__name__}"
            )

        if answer.agent_output["data_df"] is None:
            answer.gt_evaluation = Evaluation(score=0.0)
            return

        gt_df_cmp = pd.read_csv(StringIO(gt_data["data_str"]))
        gt_df_cmp.columns = [c.lower() for c in gt_df_cmp.columns]

        RetrieverEvaluator._apply_gt_alignment(answer, list(gt_df_cmp.columns))

        result_df = answer.agent_output["data_df"].copy()
        result_df.columns = [c.lower() for c in result_df.columns]

        score = compare_dataframes_iou(result_df, gt_df_cmp)
        if score < 1.0:
            n_gt = len(gt_df_cmp)
            n_model = len(result_df)
            gt_cols = set(gt_df_cmp.columns)
            model_cols = set(result_df.columns)
            if gt_cols != model_cols:
                missing = sorted(gt_cols - model_cols)
                extra = sorted(model_cols - gt_cols)
                parts = [f"GT {n_gt} rows, model {n_model} rows, IOU={score:.3f}."]
                if missing:
                    parts.append(f"Missing cols: {missing}.")
                if extra:
                    parts.append(f"Extra cols: {extra}.")
            else:
                gt_r0 = dict(zip(gt_df_cmp.columns, gt_df_cmp.iloc[0].tolist()))
                mod_r0 = (
                    dict(zip(result_df.columns, result_df.iloc[0].tolist()))
                    if n_model > 0
                    else {}
                )

        answer.gt_evaluation = Evaluation(score=score)
        return
