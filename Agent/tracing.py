from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, Optional

import pandas as pd


# Optional tracing/instrumentation (Phoenix / OpenInference)
try:
    from phoenix.otel import register as phoenix_register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.trace import StatusCode
    _PHOENIX_AVAILABLE = True
except Exception:  # pragma: no cover - tracing is optional
    StatusCode = None  # type: ignore
    _PHOENIX_AVAILABLE = False
    phoenix_register = None  # type: ignore
    LangChainInstrumentor = None  # type: ignore


_TRACE_TEXT_LIMIT = 1200
_TRACE_LIST_LIMIT = 8
_TRACE_DICT_LIMIT = 20

def _truncate_trace_text(value: Any, limit: int = _TRACE_TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _summarize_dataframe(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None:
        return {"present": False}
    columns = [str(col) for col in df.columns[:_TRACE_LIST_LIMIT]]
    summary: Dict[str, Any] = {
        "present": True,
        "rows": int(len(df.index)),
        "columns": columns,
        "column_count": int(len(df.columns)),
    }
    if len(df.columns) > _TRACE_LIST_LIMIT:
        summary["columns_truncated"] = True
    return summary


def _summarize_state_for_trace(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not state:
        return {}
    summary: Dict[str, Any] = {
        "prompt": _truncate_trace_text(state.get("prompt", "")),
        "visualization_goal": _truncate_trace_text(state.get("visualization_goal", "")),
        "tool_choice": str(state.get("tool_choice", "")),
        "answer_count": len(state.get("answer", []) or []),
        "has_error": bool(state.get("error")),
        "sql_query": _truncate_trace_text(state.get("sql_query", "")),
        "chart_config": state.get("chart_config"),
        "dataframe": _summarize_dataframe(state.get("data_df")),
    }
    data_text = state.get("data", "")
    if data_text:
        summary["data_preview"] = _truncate_trace_text(data_text)
    cached = state.get("cached_step_results") or {}
    if cached:
        summary["cached_steps"] = sorted(str(key) for key in cached.keys())[:_TRACE_LIST_LIMIT]
        summary["cached_step_count"] = len(cached)
    run_id = state.get("run_id")
    if run_id:
        summary["run_id"] = str(run_id)
    return summary


def _summarize_result_for_trace(result: Any) -> Any:
    if isinstance(result, pd.DataFrame):
        return _summarize_dataframe(result)
    if not isinstance(result, dict):
        return _truncate_trace_text(result)

    summary: Dict[str, Any] = {
        "keys": sorted(str(key) for key in result.keys())[:_TRACE_DICT_LIMIT],
        "tool_choice": str(result.get("tool_choice", "")),
        "answer_count": len(result.get("answer", []) or []),
        "has_error": bool(result.get("error")),
        "error": _truncate_trace_text(result.get("error", "")),
        "sql_query": _truncate_trace_text(result.get("sql_query", "")),
        "chart_config": result.get("chart_config"),
        "dataframe": _summarize_dataframe(result.get("data_df")),
    }
    answers = result.get("answer", []) or []
    if answers:
        summary["latest_answer"] = _truncate_trace_text(answers[-1])
    data_text = result.get("data", "")
    if data_text:
        summary["data_preview"] = _truncate_trace_text(data_text)
    for key in ("_temperature", "_top_p", "_top_k", "_run_idx", "_best_idx", "_gt_score"):
        if key in result:
            summary[key] = result[key]
    if "_all_scores" in result:
        summary["_all_scores"] = [float(score) for score in result["_all_scores"][:_TRACE_LIST_LIMIT]]
    return summary


class TracingHelper:
    """Best-effort helper for Phoenix/OpenInference tracing."""

    def __init__(self, tracer=None) -> None:
        self.tracer = tracer

    @property
    def enabled(self) -> bool:
        return self.tracer is not None

    def _normalize_attributes(self, attributes: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        if not attributes:
            return normalized
        for key, value in attributes.items():
            if value is None:
                continue
            if isinstance(value, (str, bool, int, float)):
                normalized[str(key)] = value
            elif isinstance(value, (list, tuple)):
                normalized[str(key)] = [_truncate_trace_text(item, 200) for item in value[:_TRACE_LIST_LIMIT]]
            else:
                normalized[str(key)] = _truncate_trace_text(value, 400)
        return normalized

    def set_attributes(self, span, attributes: Optional[Dict[str, Any]]) -> None:
        if span is None or not attributes:
            return
        try:
            normalized = self._normalize_attributes(attributes)
            if normalized:
                span.set_attributes(normalized)  # type: ignore[attr-defined]
        except Exception:
            pass

    def set_input(self, span, value: Any) -> None:
        if span is None or value is None:
            return
        try:
            if hasattr(span, "set_input"):
                span.set_input(value)  # type: ignore[attr-defined]
            else:
                self.set_attributes(span, {"input": _truncate_trace_text(json.dumps(value, default=str))})
        except Exception:
            pass

    def set_output(self, span, value: Any) -> None:
        if span is None or value is None:
            return
        try:
            if hasattr(span, "set_output"):
                span.set_output(value)  # type: ignore[attr-defined]
            else:
                self.set_attributes(span, {"output": _truncate_trace_text(json.dumps(value, default=str))})
        except Exception:
            pass

    def record_exception(self, span, exc: Exception) -> None:
        if span is None:
            return
        try:
            if hasattr(span, "record_exception"):
                span.record_exception(exc)  # type: ignore[attr-defined]
            self.set_attributes(
                span,
                {
                    "error.type": type(exc).__name__,
                    "error.message": _truncate_trace_text(exc),
                },
            )
        except Exception:
            pass

    def set_status_ok(self, span) -> None:
        if span is None or StatusCode is None:
            return
        try:
            span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
        except Exception:
            pass

    def set_status_error(self, span, exc: Exception) -> None:
        if span is None or StatusCode is None:
            return
        try:
            span.set_status(StatusCode.ERROR, str(exc))  # type: ignore[attr-defined]
        except Exception:
            try:
                span.set_status(StatusCode.ERROR)  # type: ignore[attr-defined]
            except Exception:
                pass

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

        span_cm = None
        span = None
        try:
            kwargs = {}
            if kind:
                kwargs["openinference_span_kind"] = kind
            span_cm = self.tracer.start_as_current_span(name, **kwargs)  # type: ignore[attr-defined]
            span = span_cm.__enter__()
            self.set_attributes(span, attributes)
            self.set_input(span, input_data)
        except Exception:
            yield None
            return

        try:
            yield span
            self.set_status_ok(span)
        except Exception as exc:
            self.record_exception(span, exc)
            self.set_status_error(span, exc)
            raise
        finally:
            if span_cm is not None:
                try:
                    span_cm.__exit__(None, None, None)
                except Exception:
                    pass

