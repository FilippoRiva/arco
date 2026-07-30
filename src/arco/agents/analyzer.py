from typing import TYPE_CHECKING

import pandas as pd

from arco.core import Agent, AgentException
from arco.evaluators import AnalyzerEvaluator

if TYPE_CHECKING:
    from arco.core import LLM, Answer, Evaluator, State

import logging

logger = logging.getLogger(__name__)


class Analyzer(Agent):
    _ANALYSE_DATA_PROMPT = """You are a professional data analyst providing insights from query results.

## TASK
Answer the user's question based ONLY on the provided data.

## USER QUESTION
{prompt}

## AVAILABLE DATA
This data was retrieved using the SQL query: 
{sql_query}

Data:
{data}

## INSTRUCTIONS
1. Examine the data carefully to understand what information is available
2. Identify the key insights that directly answer the user's question
3. Provide a concise, specific answer (2-3 sentences maximum)
4. Use actual numbers and facts from the data
5. Do NOT speculate or make assumptions beyond what the data shows
6. If the data doesn't fully answer the question, state what you can determine from the available data
7. Do NOT provide textual visualizations even when the users asks to. Your analysis should be text focused and should ignore visualization requests.

## EXAMPLES

Example 1 - Good answer (factual, concise):
    Question: "What were the total sales in November 2021? Then provide a visualization containing the top 5 transactions."
    Data: Shows 45 rows with Revenue column summing to $1,234,567
    Answer: "Based on the data, total sales in November 2021 were $1,234,567 across 45 transactions."

Example 2 - Bad answer (do NOT do this):
    Question: "What were the total sales in November 2021? Then provide a visualization containing the top 5 transactions."
    Data: Shows 45 rows with Revenue column summing to $1,234,567
    Bad Answer: "Sales were strong in November, likely due to holiday shopping. This trend probably continued into December and suggests the company is performing well."
    Why this is bad: Adds speculation ("likely due to holiday shopping"), makes assumptions beyond the data ("trend continued"), does not state the actual number.

## OUTPUT FORMAT
Provide a direct, concise answer in natural language (2-3 sentences). Focus only on facts from the data.
"""

    def __init__(self):
        super().__init__()

    @property
    def evaluator(self) -> Evaluator:
        return AnalyzerEvaluator()

    def core(self, state: State, llm: LLM) -> State:
        last_retriever_answer: Answer | None = state.get_last_answer("Retriever")
        if (
            last_retriever_answer is None
            or "data_str" not in last_retriever_answer.agent_output
            or "sql_query" not in last_retriever_answer.agent_output
        ):
            logger.error(
                f"Missing dependencies for analysis from retriever output: last_ret_answer:{last_retriever_answer}, last_retriever_output:{last_retriever_answer.agent_output}"
            )
            raise AgentException(missing_dependencies_from="Retriever")
        enriched_data = _enrich_data_with_stats(
            last_retriever_answer.agent_output["data_str"]
        )
        formatted_prompt = Analyzer._ANALYSE_DATA_PROMPT.format(
            data=enriched_data,
            prompt=state.prompt,
            sql_query=last_retriever_answer.agent_output["sql_query"],
        )
        result = llm.invoke(formatted_prompt)
        logger.info(
            f"Analysis result (logprobs : {len(result.logprobs) > 0}): {result.text}"
        )
        return self.answer(
            state,
            message=f"{result.text}",
            output={"analysis": result.text},
            logprobs=result.logprobs,
        )


def _enrich_data_with_stats(data_csv: str | None) -> str:
    """Append pre-computed numeric statistics to the CSV data string.

    LLMs are unreliable at mental arithmetic over many rows.  Pre-computing
    sum / min / max / count for every numeric column and appending them as a
    summary block lets the LLM read the answer directly instead of deriving it.
    """
    if not data_csv or not data_csv.strip():
        return data_csv if data_csv else ""
    import io

    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=io.StringIO(data_csv))  # type: ignore
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        logger.debug("No numeric columns found. Skipping data enrichment.")
        return data_csv
    lines = [
        "\n--- Pre-computed Statistics (use these exact values) ---",
        f"Total rows: {len(df)}",
    ]
    for col in num_cols:
        s = df[col]
        lines.append(
            f"{col}: sum={round(s.sum(), 2)}, min={round(s.min(), 2)}, "
            f"max={round(s.max(), 2)}, mean={round(s.mean(), 2)}"
        )
    logger.debug(f"Adding enrichment statistics : {' | '.join(lines)}")
    return data_csv + "\n".join(lines)


__all__ = ["Analyzer"]
