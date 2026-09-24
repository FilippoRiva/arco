"""General LLM-as-a-judge evaluator for tool-using agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from arco.core import Answer, Evaluation, Evaluator, State, get_llm

logger = logging.getLogger(__name__)


class ToolUseEvaluator(Evaluator):
    """Evaluate a tool-using agent against its declared role.

    The evaluator deliberately does not implement batch evaluation. Each
    candidate is judged independently by ``_eval``. Ground-truth evaluation
    compares the complete serialized output of a generated reference agent
    with the complete output of the current agent through an LLM judge.
    """

    def __init__(self, role: str):
        if not role.strip():
            raise ValueError("ToolUseEvaluator role cannot be empty")
        self.role = role

    @staticmethod
    def _answer_payload(answer: Answer, role: str) -> dict[str, Any]:
        """Return the complete agent output used by both judging paths."""
        return {
            "agent_id": str(answer.agent_id),
            "role": role,
            "message": answer.message,
            "output": answer.agent_output,
            "error": answer.error,
            "thinking": answer.thinking,
        }

    @staticmethod
    def _score(parsed: dict[str, Any]) -> float:
        score = parsed.get("score") if isinstance(parsed, dict) else None
        if isinstance(score, dict):
            score = score.get("value")
        try:
            return max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return 0.0

    def _judge_prompt(
        self,
        *,
        candidate: dict[str, Any],
        reference: dict[str, Any] | None = None,
    ) -> str:
        reference_section = ""
        if reference is not None:
            reference_section = f"""
## REFERENCE AGENT OUTPUT
Use this output as the ground-truth reference. Do not require literal
string equality; judge whether the candidate accomplishes the same task.
```json
{json.dumps(reference, indent=2, ensure_ascii=False, default=str)}
```
"""

        return f"""You are evaluating an agentic tool-using system.

## AGENT ROLE
{self.role}

## EVALUATION CRITERIA
Evaluate the complete output, including any tool calls and their results.
Consider:
- adherence to the declared role;
- correctness and usefulness of the final response;
- completeness relative to the task;
- whether tool calls were appropriate and used effectively;
- whether the output is internally consistent and free of avoidable errors.
{reference_section}
## CANDIDATE AGENT OUTPUT
```json
{json.dumps(candidate, indent=2, ensure_ascii=False, default=str)}
```

Return ONLY valid JSON in this format:
{{
  "score": <number from 0.0 to 1.0>,
  "reasoning": "brief explanation"
}}
"""

    def _eval(
        self,
        state: State,
        judge_provider: str,
        judge_model: str,
        llm_accumulator=None,
    ) -> Evaluation:
        answer = state.get_last_answer()
        if answer is None:
            return Evaluation(score=0.0, success=False)

        llm = get_llm(
            provider=judge_provider,
            model=judge_model,
            llm_accumulator=llm_accumulator,
        )
        prompt = self._judge_prompt(
            candidate=self._answer_payload(answer, self.role)
        )
        response = llm.invoke(prompt)
        score = self._score(response.extract_json())
        return Evaluation(score=score, success=True)

    def _batch_eval(self, states: list[State]) -> list[Evaluation] | None:
        return None

    def _gt_eval(
        self,
        answer: Answer,
        gt_data: dict,
        judge_provider: str,
        judge_model: str,
    ) -> Evaluation:
        llm = get_llm(provider=judge_provider, model=judge_model)
        candidate = self._answer_payload(answer, self.role)
        reference = gt_data
        prompt = self._judge_prompt(candidate=candidate, reference=reference)
        response = llm.invoke(prompt)
        score = self._score(response.extract_json())
        logger.debug("Tool-use ground-truth evaluation score=%s", score)
        return Evaluation(score=score, success=True)

    def extract_gt_from_answer(self, answer: Answer) -> dict[str, Any]:
        """Extract the complete reference output for future benchmark judging."""
        return self._answer_payload(answer, self.role)


__all__ = ["ToolUseEvaluator"]
