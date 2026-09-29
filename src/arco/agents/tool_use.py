"""General-purpose agent with provider-native tool calling."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from arco.core.agent import Agent
from arco.core.answer import AnswerDraft
from arco.core.evaluator import Evaluator
from arco.evaluators import ToolUseEvaluator

if TYPE_CHECKING:
    from arco.core.llm_tools import LLM
    from arco.core.state import State

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
        logger.debug(
            "%s initialized with %d tool(s), max_tool_iterations=%d",
            self.name,
            len(self.tools),
            self.max_tool_iterations,
        )

    @staticmethod
    def _tool_name(tool: ToolLike) -> str:
        return str(getattr(tool, "name", getattr(tool, "__name__", "")))

    @property
    def evaluator(self) -> Evaluator:
        return ToolUseEvaluator(role=self.role)

    def _execute_tool(self, name: str, args: Any) -> tuple[str, bool]:
        tool = self._tools_by_name.get(name)
        if tool is None:
            logger.debug("%s requested unknown tool %s", self.name, name)
            return f"Unknown tool: {name}", False

        argument_summary = (
            f"keys={list(args)}" if isinstance(args, dict) else type(args).__name__
        )
        logger.debug("%s executing tool %s (%s)", self.name, name, argument_summary)
        try:
            if isinstance(tool, BaseTool):
                result = tool.invoke(args)
            elif isinstance(args, dict):
                result = tool(**args)
            else:
                result = tool(args)
            result_text = _json_text(result)
            logger.debug(
                "%s tool %s completed successfully (result length: %d characters)",
                self.name,
                name,
                len(result_text),
            )
            return result_text, True
        except Exception as exc:  # Tools must return errors to the model.
            logger.exception("%s tool %s failed", self.name, name)
            return f"Tool {name} failed: {exc!s}", False

    def core(self, state: State, llm: LLM) -> AnswerDraft:
        context = [
            {
                "agent": str(answer.agent_id),
                "message": answer.message,
                # Agent instructions and tool traces are internal execution
                # details, not useful downstream context. Keep the actual
                # result fields so specialist agents can build on prior work.
                "output": {
                    key: value
                    for key, value in answer.agent_output.items()
                    if key not in {"role", "tool_calls"}
                },
                "error": answer.error,
            }
            for answer in state.answers
        ]
        user_content = (
            f"USER REQUEST:\n{state.prompt}\n\n"
            "WORKFLOW CONTEXT FROM PREVIOUS AGENTS:\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}"
        )
        logger.debug(
            "%s starting tool-use turn (prompt length: %d, prior answers: %d, tools: %d)",
            self.name,
            len(state.prompt),
            len(context),
            len(self.tools),
        )
        messages: list[Any] = [
            SystemMessage(content=self.role),
            HumanMessage(content=user_content),
        ]
        tool_trace: list[dict[str, Any]] = []
        final_response: AIMessage | None = None

        bound_llm = llm.bind_tools(self.tools)
        for iteration in range(self.max_tool_iterations):
            logger.debug(
                "%s invoking model for tool-use iteration %d/%d (%d message(s))",
                self.name,
                iteration + 1,
                self.max_tool_iterations,
                len(messages),
            )
            response = bound_llm.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError(
                    f"Tool-use model returned {type(response).__name__}, expected AIMessage"
                )
            final_response = response
            tool_calls = response.tool_calls
            messages.append(response)
            logger.debug(
                "%s received %d tool call(s) on iteration %d",
                self.name,
                len(tool_calls),
                iteration + 1,
            )

            if not tool_calls:
                logger.debug(
                    "%s completed without further tool calls after iteration %d",
                    self.name,
                    iteration + 1,
                )
                break

            for call_index, tool_call in enumerate(tool_calls):
                name = str(tool_call.get("name", ""))
                args = tool_call.get("args", {})
                tool_call_id = str(
                    tool_call.get("id") or f"tool_call_{iteration}_{call_index}"
                )
                logger.debug(
                    "%s executing tool call %d/%d: %s (id=%s)",
                    self.name,
                    call_index + 1,
                    len(tool_calls),
                    name,
                    tool_call_id,
                )
                self.emit_event(
                    "tool_started",
                    message=f"Calling {name or 'unknown tool'}",
                    data={
                        "tool": name,
                        "args": args,
                        "iteration": iteration + 1,
                        "call_index": call_index + 1,
                    },
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
            logger.debug(
                "%s reached the tool iteration limit (%d); requesting final synthesis",
                self.name,
                self.max_tool_iterations,
            )
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
        logger.debug(
            "%s produced final response (length: %d characters, tool calls: %d)",
            self.name,
            len(answer.text),
            len(tool_trace),
        )
        output = {
            "role": self.role,
            "response": answer.text,
            "tool_calls": tool_trace,
        }
        return AnswerDraft(
            message=answer.text,
            output=output,
        )


__all__ = ["ToolLike", "ToolUseAgent"]
