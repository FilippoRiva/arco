from arco import llm_tools
from arco.llm_tools import CoTRefiner
from arco.core.agent import AgentException
from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.language_models import BaseChatModel

from arco.core import Agent, Answer, AgentType
from arco.evaluators import AnalyzerEvaluator

if TYPE_CHECKING:
    from arco.core import State, AgentConfig, Evaluator
    from arco.tracing import TracingHelper


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

    ## CHAIN OF THOUGHT REASONING
    Before answering, think step by step:

    **Step 1: Understanding the Question**
    - What specific information is the user asking for?
    - Is it asking for a single value, a comparison, a trend, or a summary?
    - What would constitute a complete answer?

    **Step 2: Examining the Data Structure**
    - How many rows of data are available?
    - What columns are present in the data?
    - What is the range or distribution of values?
    - Are there any patterns or anomalies visible?

    **Step 3: Extracting Relevant Facts**
    - Which specific values directly answer the question?
    - Do I need to perform mental calculations (sum, average, count)?
    - What are the exact numbers, dates, or categories relevant to the answer?
    - Are there any context clues (time periods, units, categories)?

    **Step 4: Verifying Completeness**
    - Does the data fully answer the user's question?
    - Is there missing information that prevents a complete answer?
    - Should I mention any limitations or caveats?

    **Step 5: Formulating the Answer**
    - How can I state the facts concisely (2-3 sentences)?
    - Am I using specific numbers from the data?
    - Am I avoiding speculation or assumptions?
    - Is my answer direct and clear?



    ## EXAMPLES WITH REASONING

    Example 1 - Good answer:
    Question: "What were the total sales in November 2021?"
    Data: Shows 45 rows with Revenue column summing to $1,234,567

    Reasoning:
    - Step 1: User wants total sales amount for a specific month
    - Step 2: 45 rows of data, Revenue column present
    - Step 3: Sum of Revenue = $1,234,567, time period = November 2021, transaction count = 45
    - Step 4: Data fully answers the question, no missing info
    - Step 5: State the total, mention the number of transactions, keep it factual

    Answer: "Based on the data, total sales in November 2021 were $1,234,567 across 45 transactions."

    Example 2 - Bad answer (do NOT do this):
    Question: "What were the total sales in November 2021?"
    Data: Shows 45 rows with Revenue column summing to $1,234,567

    Bad Answer: "Sales were strong in November, likely due to holiday shopping. This trend probably continued into December and suggests the company is performing well."

    Why this is bad:
    - Violates Step 5: Adds speculation ("likely due to holiday shopping")
    - Violates instruction 5: Makes assumptions beyond data ("trend continued")
    - Violates Step 3: Doesn't state the actual number ($1,234,567)
    - Adds interpretation not supported by data ("company performing well")

    Example 3 - Handling incomplete data:
    Question: "How do our November 2021 sales compare to the previous year?"
    Data: Shows only November 2021 data (45 rows, $1,234,567 total)

    Reasoning:
    - Step 1: User wants year-over-year comparison
    - Step 2: Only 2021 data present, no 2020 data
    - Step 3: Can extract November 2021 total = $1,234,567
    - Step 4: Cannot make comparison - missing 2020 data
    - Step 5: State what we know, acknowledge limitation

    Answer: "The available data shows November 2021 sales totaled $1,234,567 across 45 transactions. However, the dataset does not include November 2020 data, so a year-over-year comparison cannot be made."

    Example 4 - Multiple data points:
    Question: "Which product had the highest revenue?"
    Data: Shows Product_Name and Revenue for 10 products, top one is "Widget Pro" with $450,000

    Reasoning:
    - Step 1: User wants to identify top-performing product
    - Step 2: 10 rows, columns are Product_Name and Revenue
    - Step 3: Maximum Revenue = $450,000, corresponding Product_Name = "Widget Pro"
    - Step 4: Data fully answers the question
    - Step 5: State the product name and its revenue value

    Answer: "Widget Pro had the highest revenue at $450,000."

    ## OUTPUT FORMAT
    Provide a direct, concise answer in natural language (2-3 sentences). Focus only on facts from the data.
    """

    def __init__(self, trace_helper: TracingHelper, empower: bool = False):
        super().__init__(trace_helper, empower)
        self.type = AgentType.ANALYZER

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
            if last_retriever_answer is None :
                raise AgentException(missing_answer_from_type=AgentType.RETRIEVER)
            enriched_data = Analyzer._enrich_data_with_stats(last_retriever_answer.data_str)
            formatted_prompt = Analyzer._ANALYSE_DATA_PROMPT.format(
                data=enriched_data, prompt=state.prompt, sql_query=last_retriever_answer.sql_query
            )
            analysis_result = llm.invoke(formatted_prompt)
            analysis_text = str(analysis_result.content) if hasattr(analysis_result, "content") else str(analysis_result)
            logprobs = llm_tools.extract_logprobs(analysis_result)
            analyzer_config = deepcopy(state.get_agent_config(self.type))
            answer: Answer = Answer(
                agent_id=self.type,
                message=f"{analysis_text}",
                analysis=analysis_text,
                agent_config=analyzer_config,
                logprobs=logprobs
            )

            return state.add_answer(answer)

        except Exception as e:
            print(f"Error analyzing data: {str(e)}")

            answer: Answer = Answer(
                agent_id=self.type,
                message="Couldn't analyze data. Check error message for details",
                error=f"Error accessing data: {str(e)}",
                agent_config=deepcopy(state.get_agent_config(AgentType.ANALYZER)),
            )

            return state.add_answer(answer)

    def get_evaluator(self, agent_config: AgentConfig) -> Evaluator:
        return AnalyzerEvaluator(agent_config)
