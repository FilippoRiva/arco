import json
import logging
from typing import TYPE_CHECKING

from arco.core import Agent
from arco.evaluators import PlannerEvaluator

if TYPE_CHECKING:
    from arco.core import LLM, Evaluator, State

_VALID_AGENTS = {"retriever", "analyzer", "visualizer"}

logger = logging.getLogger(__name__)


class Planner(Agent):
    _PLANNER_PROMPT = """You are a workflow planner for a data analysis pipeline.

## AVAILABLE AGENTS
- retriever: Retrieves data from the database using SQL
- analyzer: Analyzes retrieved data and provides insights
- visualizer: Generates chart code to visualize the data

## TASK
Given the user's question, decide which agents to execute and in what order.

## RULES
- retriever MUST come before analyzer or visualizer (data is needed first).
- analyzer should come before visualizer if both are needed.
- visualizer is ONLY needed if the user explicitly asks for a chart or graph.
- For simple factual questions, retriever + analyzer is sufficient.
- For raw data requests, just retriever.
- Include all needed agents. Do not skip necessary steps.

## EXAMPLES

Question: "Show me a bar chart of monthly sales by region"
Plan: ["retriever", "analyzer", "visualizer"]

Question: "What were the total sales in 2022?"
Plan: ["retriever", "analyzer"]

## USER QUESTION
{prompt}

## OUTPUT FORMAT
Return ONLY a JSON array of agent names in execution order.
Choose from: "retriever", "analyzer", "visualizer"
No explanations. No markdown. Just the JSON array.
"""

    _PLANNER_REROUTE_PROMPT = """An error occurred during execution of {last_agent}.

## ERROR
{error}

## ORIGINAL QUESTION
{prompt}

## REMAINING PLAN (not yet executed)
{remaining}

## TASK
Decide whether to continue or abort. You may skip the failed agent if the error is recoverable,
or output [] to end the workflow.

## OUTPUT FORMAT
Return ONLY a JSON array of the remaining agent names to execute, or [] to abort.
Choose from: "retriever", "analyzer", "visualizer"
No explanations. No markdown. Just the JSON array.
"""

    def __init__(self):
        super().__init__()

    @property
    def evaluator(self) -> Evaluator:
        return PlannerEvaluator()

    def core(self, state: State, llm: LLM) -> State:
        last_planner = state.get_last_answer(self.type)

        if last_planner is None:
            # --- FIRST INVOCATION: generate full plan from LLM ---
            formatted = self._PLANNER_PROMPT.format(prompt=state.prompt)
            response = llm.invoke(formatted)
            plan = response.extract_json_list()
            if not plan:
                plan = ["retriever", "analyzer"]
            else:
                plan = [choice.lower() for choice in plan]

            choice = plan[0].capitalize()
            remaining = plan[1:]

            logger.info(f"Choice: {choice}")

            return self.answer(
                state,
                message=f"Plan: {', '.join(a.capitalize() for a in plan)}",
                output={"agent_choice": choice, "plan": remaining},
                logprobs=response.logprobs,
            )

        # --- SUBSEQUENT INVOCATIONS: consume from plan ---
        remaining = list(last_planner.agent_output.get("plan", []))

        # Check last non-Planner answer for errors
        last_error = None
        last_agent_name = None
        for ans in reversed(state.answers):
            if ans.agent_id != self.type:
                last_agent_name = ans.agent_id
                if ans.error:
                    last_error = ans.error
                break

        if last_error:
            formatted = self._PLANNER_REROUTE_PROMPT.format(
                last_agent=last_agent_name,
                error=last_error,
                prompt=state.prompt,
                remaining=json.dumps(remaining),
            )
            response = llm.invoke(formatted)
            new_plan = response.extract_json_list()
            remaining = new_plan if new_plan else []

        if len(state.answers) > 10:
            remaining = []

        if not remaining:
            logger.info("Workflow complete")
            return self.answer(
                state,
                message="Workflow complete",
                output={"agent_choice": "End", "plan": []},
            )

        choice = remaining[0].capitalize()
        logger.info(f"Choice from previous plan: {choice}")
        return self.answer(
            state,
            message=f"Next: {choice}",
            output={"agent_choice": choice, "plan": remaining[1:]},
        )


__all__ = ["Planner"]
