from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from typing_extensions import NotRequired, TypedDict


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)

class State(TypedDict):
    prompt: str
    data: Optional[str]
    data_df: NotRequired[Optional[pd.DataFrame]]
    answer: List[str]
    visualization_goal: Optional[str]
    chart_config: Optional[dict]
    tool_choice: NotRequired[str]
    error: NotRequired[str]
    sql_query: Optional[str]
    # Per-step configuration and caching (Phase 1)
    agent_config: NotRequired[Optional[Dict]]  # AgentConfig as dict (for state flow)
    run_id: NotRequired[Optional[str]]  # Unique identifier for this execution
    cached_step_results: NotRequired[Optional[Dict]]  # Pre-loaded results from similar past runs
    # Profiling (accumulated across steps)
    _step_timings_sec: NotRequired[Optional[Dict]]
    _step_eval_scores: NotRequired[Optional[Dict]]
    # Ground truth scores per step (set by _run_gt_eval, propagated to final result)
    _gt_scores_per_step: NotRequired[Optional[Dict]]

