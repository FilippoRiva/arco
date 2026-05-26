import json
from contextlib import contextmanager
from typing import Any, Dict, Optional, TYPE_CHECKING
import os

import pandas as pd
from openinference.instrumentation import OITracer

if TYPE_CHECKING:
    from arco.core import Answer, State

from phoenix.otel import register as phoenix_register
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.trace import StatusCode

_TRACE_TEXT_LIMIT = 1200
_TRACE_LIST_LIMIT = 8
_TRACE_DICT_LIMIT = 20

def init_tracing(endpoint : str | None) -> None:
    phoenix_api_key = os.getenv("PHOENIX_API_KEY")
    if phoenix_api_key:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"
    if endpoint:
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint

def get_tracer(project_name : str | None) -> TracingHelper:
    endpoint : str | None= os.environ.get("PHOENIX_COLLECTOR_ENDPOINT") or None
    tracer_provider = phoenix_register(
        project_name=project_name or "ARCO",
        endpoint=(endpoint or "https://app.phoenix.arize.com/v1/traces"),
    )
    LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)
    return TracingHelper(tracer_provider.get_tracer(__name__))


def truncate_trace_text(value: Any, limit: int = _TRACE_TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else str(value)
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>" if len(text) <= limit else text


def summarize_dataframe(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None:
        return {"present": False}
    columns = [col for col in df.columns[:_TRACE_LIST_LIMIT]]
    summary: Dict[str, Any] = {
        "present": True,
        "rows": len(df.index),
        "columns": columns,
        "column_count": len(df.columns),
    }
    if len(df.columns) > _TRACE_LIST_LIMIT:
        summary["columns_truncated"] = True
    return summary


def _summarize_state_for_trace(state: State) -> Dict[str, Any]:
    if not state:
        return {}

    # Build a summary for state
    summary: Dict[str, Any] = {
        "prompt": truncate_trace_text(state.prompt),
        "visualization_goal": truncate_trace_text(state.visualization_goal),
        "answer_count": len(state.answers),
        "run_id": state.run_id,
    }

    last_answer: Answer | None = state.answers[-1] if state.answers else None
    if last_answer:
        summary.update({
            "tool_choice": str(last_answer.agent_choice),
            "has_error": bool(last_answer.error),
            "error": last_answer.error,
            "sql_query": truncate_trace_text(last_answer.sql_query),
            "chart_config": last_answer.chart_config,
            "latest_answer": truncate_trace_text(last_answer)
        })

    from arco.core import AgentType
    last_retriever_answer: Answer | None = state.get_last_answer(AgentType.RETRIEVER)
    if last_retriever_answer:
        summary.update({
            "dataframe": summarize_dataframe(last_retriever_answer.data_df),
            "data_preview": truncate_trace_text(last_retriever_answer.data_str),
        })

    cached = state.cached_results
    if cached:
        summary["cached_steps"] = sorted(str(key) for key in cached.keys())[:_TRACE_LIST_LIMIT]
        summary["cached_step_count"] = len(cached)
    return summary


def _summarize_for_trace(result: State, additional_logging: Dict[str, Any] | None = None) -> Dict[str, Any]:
    summary = _summarize_state_for_trace(result)
    if additional_logging:
        summary.update(additional_logging)
        if "_all_scores" in additional_logging:
            summary["_all_scores"] = [float(score) for score in additional_logging["_all_scores"][:_TRACE_LIST_LIMIT]]
    return summary


def _normalize_attributes(attributes: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    if not attributes:
        return normalized
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            # pyrefly: ignore [unnecessary-type-conversion]
            normalized[str(key)] = value
        elif isinstance(value, (list, tuple)):
            # pyrefly: ignore [unnecessary-type-conversion]
            normalized[str(key)] = [truncate_trace_text(item, 200) for item in value[:_TRACE_LIST_LIMIT]]
        else:
            # pyrefly: ignore [unnecessary-type-conversion]
            normalized[str(key)] = truncate_trace_text(value, 400)
    return normalized


def set_attributes(span, attributes: Optional[Dict[str, Any]]) -> None:
    if span is None or not attributes:
        return
    try:
        normalized = _normalize_attributes(attributes)
        if normalized:
            span.set_attributes(normalized)  # type: ignore[attr-defined]
    except Exception:
        pass


def set_status_ok(span) -> None:
    if span is None or StatusCode is None:
        return
    try:
        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
    except Exception:
        pass


def set_status_error(span, exc: Exception) -> None:
    if span is None or StatusCode is None:
        return
    try:
        span.set_status(StatusCode.ERROR, str(exc))  # type: ignore[attr-defined]
    except Exception:
        try:
            span.set_status(StatusCode.ERROR)  # type: ignore[attr-defined]
        except Exception:
            pass


def record_exception(span, exc: Exception) -> None:
    if span is None:
        return
    try:
        if hasattr(span, "record_exception"):
            span.record_exception(exc)  # type: ignore[attr-defined]
        set_attributes(
            span,
            {
                "error.type": type(exc).__name__,
                "error.message": truncate_trace_text(exc),
            },
        )
    except Exception:
        pass


def set_output(span, value: Any) -> None:
    if span is None or value is None:
        return
    try:
        if hasattr(span, "set_output"):
            span.set_output(value)  # type: ignore[attr-defined]
        else:
            set_attributes(span, {"output": truncate_trace_text(json.dumps(value, default=str))})
    except Exception:
        pass


def set_input(span, value: Any) -> None:
    if span is None or value is None:
        return
    try:
        if hasattr(span, "set_input"):
            span.set_input(value)  # type: ignore[attr-defined]
        else:
            set_attributes(span, {"input": truncate_trace_text(json.dumps(value, default=str))})
    except Exception:
        pass


class TracingHelper:
    """Best-effort helper for Phoenix/OpenInference tracing."""

    def __init__(self, tracer : OITracer | None= None) -> None:
        self.tracer = tracer

    @property
    def enabled(self) -> bool:
        return self.tracer is not None

    @contextmanager
    def start_span(
            self,
            name: str,
            *,
            kind: Optional[str] = None,
            attributes: Optional[Dict[str, Any]] = None,
            input_data: Any = None,
    ):
        if not self.enabled:
            yield None
            return

        try:
            kwargs = {}
            if kind:
                kwargs["openinference_span_kind"] = kind
            span_cm = self.tracer.start_as_current_span(name, **kwargs)  # type: ignore[attr-defined]
            span = span_cm.__enter__()
            set_attributes(span, attributes)
            set_input(span, input_data)
        except Exception:
            yield None
            return

        try:
            yield span
            set_status_ok(span)
        except Exception as exc:
            record_exception(span, exc)
            set_status_error(span, exc)
            raise
        finally:
            if span_cm is not None:
                try:
                    span_cm.__exit__(None, None, None)
                except Exception:
                    pass
