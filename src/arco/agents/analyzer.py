from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.language_models import BaseChatModel

from arco.core import Agent, AgentType, Answer, llm_tools
from arco.core.agent import AgentException
from arco.evaluators import AnalyzerEvaluator

if TYPE_CHECKING:
    from arco.core import Evaluator, State
    from arco.core.llm_tools import CoTRefiner


class Analyzer(Agent):
    _ANALYSE_DATA_PROMPT = """You are a professional data analyst providing insights from query results.

## TASK
Answer the user's question based ONLY on the provided data.

## USER QUESTION
{prompt}

## AVAILABLE DATA
This data was retrieved using the SQL query: {sql_query}

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

    @staticmethod
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
            return data_csv
        lines = ["\n--- Pre-computed Statistics (use these exact values) ---", f"Total rows: {len(df)}"]
        for col in num_cols:
            s = df[col]
            lines.append(
                f"{col}: sum={round(s.sum(), 2)}, min={round(s.min(), 2)}, "
                f"max={round(s.max(), 2)}, mean={round(s.mean(), 2)}"
            )
        return data_csv + "\n".join(lines)

    def core(self, state: State, llm: BaseChatModel | CoTRefiner) -> State:
        """Core analysis logic - LLM-based data analysis.

        Args:
            state: Conversation state; should include 'data' and 'prompt'.
            llm: LLM instance for analysis.

        Returns:
            Updated state with analysis appended to 'answer'.
        """
        try:
            last_retriever_answer: Answer | None = state.get_last_answer(AgentType.RETRIEVER)
            if last_retriever_answer is None:
                raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)
            enriched_data = Analyzer._enrich_data_with_stats(last_retriever_answer.agent_output['data_str'])
            formatted_prompt = Analyzer._ANALYSE_DATA_PROMPT.format(
                data=enriched_data, prompt=state.prompt, sql_query=last_retriever_answer.agent_output['sql_query']
            )
            analysis_result = llm.invoke(formatted_prompt)
            analysis_text = str(analysis_result.content) if hasattr(analysis_result, "content") else str(
                analysis_result)
            logprobs = llm_tools.extract_logprobs(analysis_result)
            analyzer_config = deepcopy(state.get_agent_config(self.type))
            answer: Answer = Answer(
                agent_id=self.type,
                message=f"{analysis_text}",
                agent_output={"analysis": analysis_text},
                agent_config=analyzer_config,
                logprobs=logprobs
            )

            return state.add_answer(answer)

        except Exception as e:
            print(f"Error analyzing data: {e!s}")

            answer: Answer = Answer(
                agent_id=self.type,
                message="Couldn't analyze data. Check error message for details",
                error=f"Error accessing data: {e!s}",
                agent_config=deepcopy(state.get_agent_config(AgentType.ANALYZER)),
            )

            return state.add_answer(answer)

    @staticmethod
    def get_evaluator() -> Evaluator:
        return AnalyzerEvaluator()
