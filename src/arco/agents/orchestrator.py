import difflib
from copy import deepcopy
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco.core import Agent, Answer, AgentType
from arco.llm_tools import CoTRefiner

if TYPE_CHECKING:
    from arco.core import State
    from arco.tracing import TracingHelper


class Orchestrator(Agent):
    _ORCHESTRATOR_PROMPT = """You are a workflow orchestrator managing a data analysis pipeline.
    
    ## AVAILABLE AGENTS
    - {AgentType.RETRIEVER.value}: Retrieves data from the database using SQL
    - {AgentType.ANALYZER.value}: Analyzes retrieved data and provides insights
    - {AgentType.VISUALIZER.value}: Generates chart code to visualize the data
    - End: Completes the workflow

    ## DECISION RULES (CRITICAL - Follow in order)
    1. Data prerequisite: Must run retriever BEFORE analyzer or visualizer
    2. No repetition: NEVER select an agent that has already been used
    3. Completion criteria: Select 'end' when:
       - The visualization has been generated
       - All relevant agents for the user's request have been executed

    ## DECISION FLOWCHART
    Start → Has data? No → {AgentType.RETRIEVER.value}
                  ↓ Yes
              Already analyzed? No → {AgentType.ANALYZER.value}
                  ↓ Yes
              Need visualization? Yes → {AgentType.VISUALIZER.value}
                  ↓ No/Done
              end
              
    ## CURRENT STATE
    - User's request: {prompt}
    - Answers generated so far: {answers}
    - Visualization goal: {visualization_goal}
    - Last agent used: {agent_choice}

    ## CHAIN OF THOUGHT REASONING
    Before selecting the next tool, think step by step:

    **Step 1: Analyzing User Request**
    - What is the user asking for? (data lookup, analysis, visualization, or combination)
    - Does the request explicitly or implicitly require a chart/graph?
    - Is this a simple data retrieval or complex multi-step task?

    **Step 2: Checking Current Progress**
    - What tools have already been executed? (check Last tool used)
    - Do we have data available? (check if lookup_sales_data was run)
    - How many answers have been generated? (check Answers generated so far)
    - What stage of the workflow are we in?

    **Step 3: Identifying What's Missing**
    - If no data: Need lookup_sales_data first (Rule 1)
    - If data exists but no analysis: Need analyzing_data
    - If analysis exists but user wants visualization: Need create_visualization
    - If all required steps done: Need end

    **Step 4: Applying Decision Rules**
    - Rule 1 check: Do I have data before attempting analysis/visualization?
    - Rule 2 check: Am I about to repeat a tool already used?
    - Rule 3 check: Have I completed all necessary steps (2+ answers OR all relevant tools)?

    **Step 5: Making the Decision**
    - Based on steps 1-4, which tool should execute next?
    - Does this choice follow the DECISION FLOWCHART?
    - Is this the minimum necessary step to progress toward completion?

    ## EXAMPLES WITH REASONING

    Example 1 - Initial state:
    State: prompt="Show sales data", answer=[], tool_choice=None

    Reasoning:
    - Step 1: User wants sales data (implies lookup needed)
    - Step 2: No agent executed yet, no data, no answers
    - Step 3: Missing everything - start with data retrieval
    - Step 4: Rule 1 applies - need data first, Rule 2 N/A (nothing used), Rule 3 not met (0 answers)
    - Step 5: Must start with lookup_sales_data

    Decision: {AgentType.RETRIEVER.value} (need data first)

    Example 2 - After data lookup:
    State: prompt="Show sales data", answer=[], tool_choice="retriever", data exists

    Reasoning:
    - Step 1: User wants sales data shown (implies analysis/presentation needed)
    - Step 2: retriever executed, data available, but 0 answers generated
    - Step 3: Have data, missing analysis
    - Step 4: Rule 1 satisfied (have data), Rule 2 check (can't repeat lookup), Rule 3 not met (0 answers)
    - Step 5: Next logical step is analyzing_data

    Decision: {AgentType.ANALYZER.value} (have data, now analyze)

    Example 3 - After analysis and visualization:
    State: prompt="Show sales trends", answer=["Analysis text", "Chart code"], tool_choice="visualizer"

    Reasoning:
    - Step 1: User wanted trends (implies analysis + visualization)
    - Step 2: All tools executed, 2 answers generated (analysis + chart code)
    - Step 3: Nothing missing - workflow complete
    - Step 4: Rule 1 satisfied, Rule 2 satisfied, Rule 3 MET (2+ answers generated)
    - Step 5: Should end the workflow

    Decision: End (visualization created, workflow complete)

    Example 4 - After analysis only (no viz needed):
    State: prompt="What were total sales?", answer=["Total sales were $X"], tool_choice="analyzer"

    Reasoning:
    - Step 1: User wanted a simple factual answer (no visualization implied)
    - Step 2: lookup and analysis executed, 1 answer generated
    - Step 3: Question fully answered with analysis alone
    - Step 4: Rule 1 satisfied, Rule 2 satisfied, Rule 3 check (all RELEVANT tools done)
    - Step 5: No visualization needed for this query - can end

    Decision: End (all relevant tools executed, question answered)

    Example 5 - After lookup only (viz needed):
    State: prompt="Show me a chart of monthly sales", answer=[], tool_choice="retriever", data exists

    Reasoning:
    - Step 1: User explicitly wants a chart (visualization required)
    - Step 2: Only lookup executed, data available, 0 answers
    - Step 3: Missing both analysis AND visualization
    - Step 4: Rule 1 satisfied (have data), Rule 2 satisfied (not repeating), Rule 3 not met (0 answers)
    - Step 5: Should analyze first, then visualize (follow flowchart)

    Decision: {AgentType.ANALYZER.value} (analyze before visualizing)

    ## YOUR TASK
    Based on the chain of thought reasoning above and the current state, select the next tool to execute.

    ## OUTPUT FORMAT
    Respond with ONLY the tool name: retriever, analyzer, visualizer, or end
    No explanations. Just the tool name.
    """

    def __init__(self, trace_helper: TracingHelper):
        super().__init__(trace_helper)
        self.type = AgentType.ORCHESTRATOR

    def core(self, state: State, llm: BaseChatModel | CoTRefiner) -> State:
        """Core tool decision logic - LLM-based routing.

        Args:
            state: Conversation state.
            llm: LLM instance for decision.

        Returns:
            Updated state with 'tool_choice'.
        """

        last_orchestrator_answer: Answer | None = state.get_last_answer(AgentType.ORCHESTRATOR)
        last_retriever_answer: Answer | None = state.get_last_answer(AgentType.RETRIEVER)

        decision_prompt = Orchestrator._ORCHESTRATOR_PROMPT.format(
            prompt=state.prompt,
            answers=state.stringify_answers(),
            visualization_goal=state.visualization_goal,
            agent_choice=last_orchestrator_answer.agent_choice if last_orchestrator_answer else None)

        # try:
        response_content: str = str(llm.invoke(decision_prompt).content)
        tool_choice = response_content.strip().lower()
        valid_tools = ["retriever", "analyzer", "visualizer", "end"]
        closest_match = difflib.get_close_matches(tool_choice, valid_tools, n=1, cutoff=0.6)
        matched_agent = closest_match[0] if closest_match else "retriever"

        # fallback if the agent selects analysis without data
        if matched_agent in ["analyzer", "visualizer"] and not last_retriever_answer:
            matched_agent = "retriever"

        # Anti-loop guard: if lookup already ran but returned no data (SQL error), stop
        if (last_orchestrator_answer and matched_agent == "retriever" and
                last_orchestrator_answer.agent_choice == "retriever" and last_retriever_answer and
                not last_retriever_answer.data_str):
            matched_agent = "end"

        # Override decision if reached max number of calls
        if len(state.answers) > 10:
            matched_agent = "end"

        answer = Answer(
            agent_id=self.type,
            message=f"The chosen agent is {matched_agent}",
            agent_choice=matched_agent,
            agent_config=deepcopy(state.get_agent_config(AgentType.ORCHESTRATOR))
        )
        return state.add_answer(answer)
