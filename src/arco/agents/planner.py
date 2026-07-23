import json
from copy import deepcopy
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from arco.core import Agent, Answer, AgentType, llm_tools, Evaluator

if TYPE_CHECKING:
    from arco.core.llm_tools import CoTRefiner
    from arco.core import State

_VALID_AGENTS = {"retriever", "analyzer", "visualizer"}


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

    @staticmethod
    def _parse_plan(raw: str) -> list[str]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
        try:
            plan = json.loads(raw)
            if not isinstance(plan, list):
                return []
            return [a.lower() for a in plan if isinstance(a, str) and a.lower() in _VALID_AGENTS]
        except (json.JSONDecodeError, TypeError):
            return []

    def core(self, state: State, llm: BaseChatModel | CoTRefiner) -> State:
        last_planner = state.get_last_answer(self.type)

        if last_planner is None:
            # --- FIRST INVOCATION: generate full plan from LLM ---
            formatted = self._PLANNER_PROMPT.format(prompt=state.prompt)
            response = llm.invoke(formatted)
            raw = response.content if hasattr(response, "content") else str(response)
            logprobs = llm_tools.extract_logprobs(response)

            plan = self._parse_plan(raw)
            if not plan:
                plan = ["retriever", "analyzer"]

            choice = plan[0].capitalize()
            remaining = plan[1:]

            answer = Answer(
                agent_id=self.type,
                message=f"Plan: {', '.join(a.capitalize() for a in plan)}",
                agent_output={"agent_choice": choice, "plan": remaining},
                agent_config=deepcopy(state.get_agent_config(self.type)),
                logprobs=logprobs,
            )
            return state.add_answer(answer)

        # --- SUBSEQUENT INVOCATIONS: consume from plan ---
        remaining = list(last_planner.agent_output.get("plan", []))

        # Check last non-Planner answer for errors
        last_error = None
        last_agent_name = None
        for ans in reversed(state.answers):
            if ans.agent_id != self.type:
                last_agent_name = ans.agent_id.value
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
            raw = response.content if hasattr(response, "content") else str(response)
            logprobs = llm_tools.extract_logprobs(response)

            new_plan = self._parse_plan(raw)
            remaining = new_plan if new_plan else []

        if len(state.answers) > 10:
            remaining = []

        if not remaining:
            answer = Answer(
                agent_id=self.type,
                message="Workflow complete",
                agent_output={"agent_choice": "end", "plan": []},
                agent_config=deepcopy(state.get_agent_config(self.type)),
            )
            return state.add_answer(answer)

        choice = remaining[0].capitalize()
        answer = Answer(
            agent_id=self.type,
            message=f"Next: {choice}",
            agent_output={"agent_choice": choice, "plan": remaining[1:]},
            agent_config=deepcopy(state.get_agent_config(self.type)),
        )
        return state.add_answer(answer)

    @staticmethod
    def get_evaluator() -> Evaluator:
        from arco.evaluators import PlannerEvaluator
        return PlannerEvaluator()