"""General-purpose agent with provider-native tool calling."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from arco.core import Agent, Evaluator
from arco.evaluators import ToolUseEvaluator

if TYPE_CHECKING:
    from arco.core import LLM, State

logger = logging.getLogger(__name__)

ToolLike = BaseTool | Callable[..., Any]


def _json_text(value: Any) -> str:
    """Represent a tool result as content suitable for a ToolMessage."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError, ValueError:
        return str(value)


class ToolUseAgent(Agent):
    """An agent that can autonomously call a supplied set of tools.

    The agent sends ``role`` as its system instruction and uses the workflow
    prompt as the user message. When the model emits tool calls, each call is
    executed and its result is appended as a ``ToolMessage`` before the model
    is invoked again. The loop ends when the model returns a normal response.
    If ``max_tool_iterations`` is reached, the model is reprompted without
    tools to analyze the tool calls and results collected so far.

    Tools may be LangChain ``BaseTool`` instances (including tools created
    with ``@langchain_core.tools.tool``) or ordinary callables. The same tool
    objects are passed to the provider's native ``bind_tools`` implementation.
    """

    def __init__(
        self,
        role: str,
        tools: Sequence[ToolLike] = (),
        *,
        agent_name: str | None = None,
        max_tool_iterations: int = 10,
    ) -> None:
        super().__init__(agent_name=agent_name)
        if not role.strip():
            raise ValueError("ToolUseAgent role cannot be empty")
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be at least 1")
        self.role = role
        self.tools = tuple(tools)
        self.max_tool_iterations = max_tool_iterations
        self._tools_by_name = {self._tool_name(tool): tool for tool in self.tools}

    @staticmethod
    def _tool_name(tool: ToolLike) -> str:
        return str(getattr(tool, "name", getattr(tool, "__name__", "")))

    @property
    def evaluator(self) -> Evaluator:
        return ToolUseEvaluator(role=self.role)

    def _execute_tool(self, name: str, args: Any) -> tuple[str, bool]:
        tool = self._tools_by_name.get(name)
        if tool is None:
            return f"Unknown tool: {name}", False

        try:
            if isinstance(tool, BaseTool):
                result = tool.invoke(args)
            elif isinstance(args, dict):
                result = tool(**args)
            else:
                result = tool(args)
            return _json_text(result), True
        except Exception as exc:  # Tools must return errors to the model.
            logger.exception("Tool %s failed", name)
            return f"Tool {name} failed: {exc!s}", False

    def core(self, state: State, llm: LLM) -> State:
        context = [
            {
                "agent": str(answer.agent_id),
                "message": answer.message,
                "output": answer.agent_output,
                "error": answer.error,
            }
            for answer in state.answers
        ]
        user_content = (
            f"USER REQUEST:\n{state.prompt}\n\n"
            "WORKFLOW CONTEXT FROM PREVIOUS AGENTS:\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}"
        )
        messages: list[Any] = [
            SystemMessage(content=self.role),
            HumanMessage(content=user_content),
        ]
        tool_trace: list[dict[str, Any]] = []
        final_response: AIMessage | None = None

        bound_llm = llm.bind_tools(self.tools)
        for iteration in range(self.max_tool_iterations):
            response = bound_llm.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(
                    f"Tool-use model returned {type(response).__name__}, expected AIMessage"
                )
            final_response = response
            tool_calls = response.tool_calls
            messages.append(response)

            if not tool_calls:
                break

            for call_index, tool_call in enumerate(tool_calls):
                name = str(tool_call.get("name", ""))
                args = tool_call.get("args", {})
                tool_call_id = str(
                    tool_call.get("id") or f"tool_call_{iteration}_{call_index}"
                )
                result, success = self._execute_tool(name, args)
                tool_trace.append(
                    {
                        "iteration": iteration + 1,
                        "id": tool_call_id,
                        "name": name,
                        "args": args,
                        "result": result,
                        "success": success,
                    }
                )
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call_id,
                        name=name or None,
                        status="success" if success else "error",
                    )
                )
        else:
            # The model used all available tool iterations. Give it one final
            # chance to synthesize the results already collected instead of
            # failing the whole agent execution. Binding an empty tool list
            # prevents this final analysis from starting another tool loop.
            messages.append(
                HumanMessage(
                    content=(
                        "The maximum number of tool calls has been reached. "
                        "Do not call any more tools. Analyze the tool calls and "
                        "their results already executed above, then provide the "
                        "best final answer to the user's request."
                    )
                )
            )
            analysis_llm = llm.bind_tools([])
            response = analysis_llm.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(
                    f"Tool-use model returned {type(response).__name__}, expected AIMessage"
                )
            final_response = response

        if final_response is None:
            raise RuntimeError("Tool-use agent did not receive a model response")

        from arco.core.llm_tools import LLMAnswer

        answer = LLMAnswer(final_response)
        output = {
            "role": self.role,
            "response": answer.text,
            "tool_calls": tool_trace,
        }
        return self.answer(
            state,
            message=answer.text,
            output=output,
            logprobs=answer.logprobs,
            thinking=answer.reasoning,
        )


__all__ = ["ToolLike", "ToolUseAgent"]
