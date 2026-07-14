import difflib
from copy import deepcopy
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco.core import Agent, Answer, AgentType, llm_tools

if TYPE_CHECKING:
    from arco.core.llm_tools import CoTRefiner
    from arco.core import State


class Orchestrator(Agent):
    _ORCHESTRATOR_PROMPT = """You are a workflow orchestrator managing a data analysis pipeline.
    
## AVAILABLE AGENTS
- retriever: Retrieves data from the database using SQL
- analyzer: Analyzes retrieved data and provides insights
- visualizer: Generates chart code to visualize the data
- end: Completes the workflow

## DECISION RULES (CRITICAL)
0. Error: if an error is present end the workflow execution
1. Data prerequisite: Must run retriever BEFORE analyzer or visualizer
2. No repetition: NEVER select an agent that has already been used
3. Completion criteria: Select 'end' when:
   - The visualization has been generated
   - All relevant agents for the user's request have been executed
   
## CHAIN OF THOUGHT REASONING
Before selecting the next tool, think step by step:

**Step 1: Analyzing User Request**
    - Is the user asking for a data analysis task or data visualization task?
    - Does the request explicitly or implicitly require a chart/graph?

**Step 2: Checking Current Progress**
    - What agents have already been executed? (check agents_used)
    - Do we have data available? (check if retriever has been executed)
    - Do we have an analysis available? (check if analyzer has been executed)
    - Do we have a visualization available? (check if visualizer has been executed)

**Step 3: Identifying What's Missing**
    - If we need data and no retriever has been run: retriever is needed first to query the database
    - If we need data analysis and no analyzer has been run: need analyzer
    - If we need a visualization but no visualizer has run: need visualizer
    - If all required steps done: need to end the workflow

**Step 4: Applying Decision Rules**
    - Rule 0 check: Did an error show up during execution?
    - Rule 1 check: Do I have data before attempting analysis/visualization?
    - Rule 2 check: Am I about to repeat an agent already used?
    - Rule 3 check: Have I completed all necessary steps?

**Step 5: Making the Decision**
    - Based on steps 1-4, which tool should execute next?
    - Is this the minimum necessary step to progress toward completion?

## EXAMPLES WITH REASONING

Example 1 - Initial state:
    User Prompt: 
    - prompt = "Show sales data", 
    - visualization_goal = None, 
    Current State:
    - agents_used = []
    - error = false

    Reasoning:
    - Step 1: User wants a visualization of sales data. It explicitly requires a visualization
    - Step 2: No agent executed yet 
    - Step 3: Since there's no data the retriever agent is needed
    - Step 4: Rule 1 applies - need data first, Rule 2 : N/A , Rule 3 : no
    - Step 5: Must start with retriever

    Decision: retriever

Example 2 - After data lookup:
    User Prompt: 
    - prompt = "Show sales data", 
    - visualization_goal = None, 
    Current State:
    - agents_used = ['retriever']
    - error = false

    Reasoning:
    - Step 1: User wants sales data shown (implies analysis/presentation needed)
    - Step 2: retriever executed, only the retriever query is available
    - Step 3: We have data and we need analysis for a proper visualization, missing analysis
    - Step 4: Rule 1 satisfied (retriever executed), Rule 2 check (can't repeat lookup), Rule 3 not met
    - Step 5: Next logical step is analyzer

    Decision: analyzer

Example 3 - After analysis and visualization:
    User Prompt: 
    - prompt = "Show sales trends", 
    - visualization_goal = None, 
    Current State:
    - agents_used = ['retriever', 'analyzer', 'visualizer']
    - error = false
    
Reasoning:
- Step 1: User wanted trends (implies analysis + visualization)
- Step 2: All tools executed, 2 answers generated (analysis + chart code)
- Step 3: Nothing missing - workflow complete
- Step 4: Rule 1 satisfied (retriever executed), Rule 2 check (can't repeat retriever, analyzer and visualizer), Rule 3 met (all answers given)
- Step 5: Should end the workflow

Decision: end 

Example 4 - After analysis only (no visualization needed):
    User Prompt: 
    - prompt = "What were total sales in 2022", 
    - visualization_goal = None, 
    Current State:
    - agents_used = ['retriever', 'analyzer']
    - error = false

    Reasoning:
    - Step 1: User wants a simple factual answer (no visualization implied)
    - Step 2: lookup and analysis already executed
    - Step 3: Question fully answered with analysis alone
    - Step 4: Rule 1 satisfied, Rule 2 satisfied, Rule 3 check (all RELEVANT tools done)
    - Step 5: No visualization needed for this query - can end

    Decision: end (all relevant tools executed, question answered)

Example 5 - After lookup only (visualization needed):
    User Prompt: 
    - prompt = "Show me a chart of montly sales",
    - visualization_goal = "Give me a bar chart, where x axis is months", 
    Current State:
    - agents_used = ['retriever']
    - error = false

    Reasoning:
    - Step 1: User explicitly wants a chart (visualization required)
    - Step 2: Only retriever executed so the data is available, no analysis or visualization available
    - Step 3: Missing both analysis and visualization for the task
    - Step 4: Rule 1 satisfied, Rule 2 check (do not repeat retriever), Rule 3 not met (no visualization provided)
    - Step 5: Should analyze first, then visualize

    Decision: analyzer (analyze before visualizing)
    
Example 6 - Error present:
    Current State:
    - agents_used = ['retriever']
    - error = true

    Reasoning:
    - Step 0: An error is present.
    - Highest-priority rule applies.
    - Workflow must stop immediately.

    Decision: end 


## YOUR TASK
Based on the chain of thought reasoning above and the current state, select the next agent to execute.
          
## USER PROMPT 
- prompt = {prompt}
- visualization_goal = {visualization_goal}

## CURRENT STATE
- agents_used = {agents_used}
- error = {error_is_present}

## OUTPUT FORMAT
Respond with ONLY the tool name: retriever, analyzer, visualizer, or end
No explanations. Just the agent's name."""

    def __init__(self, empower: bool = False):
        super().__init__(empower)
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
        last_visualizer_answer: Answer | None = state.get_last_answer(AgentType.VISUALIZER)

        error_is_present = (last_retriever_answer is not None and last_retriever_answer.error is not None
                            or last_visualizer_answer is not None and last_visualizer_answer.error is not None)

        decision_prompt = Orchestrator._ORCHESTRATOR_PROMPT.format(
            prompt=state.prompt,
            visualization_goal=state.visualization_goal,
            agents_used=state.get_agents_used(),
            error_is_present=error_is_present)

        # try:
        orchestrator_response = llm.invoke(decision_prompt)

        response_content: str = str(orchestrator_response.content)
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

        matched_agent = matched_agent.capitalize()

        logprobs = llm_tools.extract_logprobs(orchestrator_response)

        answer = Answer(
            agent_id=self.type,
            message=f"The chosen agent is {matched_agent}",
            agent_choice=matched_agent,
            agent_config=deepcopy(state.get_agent_config(AgentType.ORCHESTRATOR)),
            logprobs=logprobs
        )
        return state.add_answer(answer)
