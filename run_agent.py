#!/usr/bin/env python3
"""Run the DataAgent from a YAML configuration file.

Usage:
    python run_agent.py                           # uses config/run_config.yaml
    python run_agent.py config/my_config.yaml     # custom config path
"""
import sys
import os
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Agent.config import AgentConfig
from Agent.data_agent import SalesDataAgent


# ---------------------------------------------------------------------------
# Interactive run-parameter configuration
# ---------------------------------------------------------------------------

# Parameters the user can override, with (key, type, description).
_RUN_PARAMS = [
    ("prompt", str, "Natural language query"),
    ("visualization_goal", str, "Chart description (empty to skip)"),
    ("agent_mode", str, "Run mode: lookup_only | analysis | full"),
    ("run_id", str, "Run ID (empty = auto-generate)"),
    ("save_dir", str, "Output directory"),
    ("save_results", bool, "Save results to disk"),
    ("save_execution_artifacts", bool, "Save per-run execution artifacts"),
    ("enable_codecarbon", bool, "Enable CodeCarbon tracking"),
    ("interactive_config", bool, "Prompt for step params at runtime"),
]

_AGENT_PARAMS = [
    ("model", str, "LLM model name"),
    ("provider", str, "LLM provider: openai | ollama"),
    ("ollama_url", str, "Ollama server URL (ignored for openai)"),
]


def _prompt_value(name, current_value, param_type, description):
    """Prompt for a single value; return current_value on empty input."""
    while True:
        raw = input(f"  {name} [{current_value}]: ").strip()
        if not raw:
            return current_value
        try:
            if param_type is bool:
                if raw.lower() in ("true", "1", "yes", "y"):
                    return True
                if raw.lower() in ("false", "0", "no", "n"):
                    return False
                raise ValueError
            if param_type is str:
                if raw.lower() in ("none", "null", ""):
                    return None
                return raw
            return param_type(raw)
        except (ValueError, TypeError):
            print(f"    Invalid value for {name} (expected {param_type.__name__}). Try again.")


def _interactive_configure(agent_config, run_params):
    """Show YAML defaults and let the user override them one by one.

    Returns the (possibly modified) agent_config and run_params.
    """
    if not sys.stdin.isatty():
        return agent_config, run_params

    # --- Agent settings ---
    print("\n── Agent settings ──")
    for key, _, desc in _AGENT_PARAMS:
        value = getattr(agent_config, key, None)
        print(f"  {key:25s} {value}")
    print()

    choice = input("Accept agent settings? [Y/n]: ").strip().lower()
    if choice in ("n", "no"):
        for key, ptype, desc in _AGENT_PARAMS:
            current = getattr(agent_config, key)
            new_val = _prompt_value(key, current, ptype, desc)
            setattr(agent_config, key, new_val)

    # --- Run parameters ---
    print("\n── Run parameters ──")
    agent_mode = _mode_from_flags(run_params)
    display_params = _build_display_params(run_params, agent_mode)
    for key, value, _ in display_params:
        print(f"  {key:25s} {value}")
    print()

    choice = input("Accept run parameters? [Y/n]: ").strip().lower()
    if choice in ("n", "no"):
        for key, ptype, desc in _RUN_PARAMS:
            if key == "agent_mode":
                current = agent_mode
            else:
                current = run_params.get(key)
            new_val = _prompt_value(key, current, ptype, desc)
            if key == "agent_mode":
                run_params["lookup_only"] = new_val == "lookup_only"
                run_params["no_vis"] = new_val in ("lookup_only", "analysis")
            else:
                run_params[key] = new_val

    return agent_config, run_params


def _mode_from_flags(run_params):
    """Derive agent_mode string from lookup_only/no_vis flags."""
    if run_params.get("lookup_only"):
        return "lookup_only"
    if run_params.get("no_vis"):
        return "analysis"
    return "full"


def _build_display_params(run_params, agent_mode):
    """Build a list of (key, value, description) for display."""
    result = []
    for key, _, desc in _RUN_PARAMS:
        if key == "agent_mode":
            result.append((key, agent_mode, desc))
        else:
            result.append((key, run_params.get(key), desc))
    return result


def _slugify_prompt(prompt, max_len=48):
    """Return a filesystem-safe prompt slug."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", (prompt or "").strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        return "run"
    return normalized[:max_len].rstrip("_") or "run"


def _serialize_for_json(value):
    """Convert results into a JSON-safe structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(item) for item in value]

    if value.__class__.__name__ == "DataFrame" and hasattr(value, "to_dict"):
        try:
            return {
                "__dataframe__": True,
                "records": value.to_dict(orient="records"),
            }
        except Exception:
            return str(value)

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _build_execution_artifact_dir(save_root, run_id, prompt, timestamp=None):
    """Create and return a unique per-run artifact directory."""
    base_dir = Path(save_root or "./output")
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_id or "no-run-id")).strip("_") or "no-run-id"
    prompt_slug = _slugify_prompt(prompt)
    artifact_dir = base_dir / f"{ts}_{run_token}_{prompt_slug}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _write_execution_artifacts(
    *,
    config_path,
    save_root,
    prompt,
    result,
    agent_config,
    run_params,
):
    """Persist config, effective parameters, result, and metadata for one execution."""
    run_id = None
    if isinstance(result, dict):
        run_id = result.get("run_id")
    artifact_dir = _build_execution_artifact_dir(save_root, run_id, prompt)

    shutil.copy2(config_path, artifact_dir / "config_used.yaml")

    effective_run_params = dict(run_params)
    effective_run_params["prompt"] = prompt
    effective_config = {
        "agent": {
            "model": agent_config.model,
            "provider": agent_config.provider,
            "ollama_url": agent_config.ollama_url,
        },
        "steps": agent_config.to_dict(),
        "run": effective_run_params,
    }
    with open(artifact_dir / "effective_run_config.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_for_json(effective_config), f, indent=2)

    with open(artifact_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_for_json(result), f, indent=2)

    # --- profiling fields extracted from result ---
    step_timings = None
    total_run_time = None
    energy = None
    accuracy = None
    if isinstance(result, dict):
        step_timings = result.get("_step_timings_sec")
        total_run_time = result.get("_total_run_time_sec")
        energy = result.get("_energy")
        if "_gt_score" in result:
            accuracy = {
                "type": "ground_truth",
                "gt_score": result["_gt_score"],
                "all_gt_scores": result.get("_all_gt_scores"),
            }
        elif "_step_eval_scores" in result:
            accuracy = {
                "type": "eval_scores",
                "step_eval_scores": result["_step_eval_scores"],
            }

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "run_id": run_id,
        "prompt": prompt,
        "provider": agent_config.provider,
        "model": agent_config.model,
        # --- profiling ---
        "timing": {
            "total_run_time_sec": total_run_time,
            "step_timings_sec": step_timings,
        },
        "energy": energy,
        "accuracy": accuracy,
        "effective_run_params": _serialize_for_json(effective_run_params),
    }
    with open(artifact_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return artifact_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/run_config.yaml"
    if not os.path.isfile(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")
    agent_config, run_params, schema = AgentConfig.from_yaml(config_path)

    # Extract tracing config (goes to SalesDataAgent.__init__, not run())
    tracing = run_params.pop('tracing', {})

    # Interactive configuration of agent and run parameters
    agent_config, run_params = _interactive_configure(agent_config, run_params)

    # Extract prompt (positional arg to run())
    prompt = run_params.pop('prompt')
    if not prompt:
        print("Error: 'run.prompt' is required")
        sys.exit(1)

    save_execution_artifacts = run_params.pop('save_execution_artifacts', True)

    # Always use interactive provider when running from a terminal
    run_params.pop('interactive_config', None)
    parameter_provider = None
    if sys.stdin.isatty():
        from Agent.parameter_provider import TerminalProvider
        parameter_provider = TerminalProvider()

    # Build agent
    agent = SalesDataAgent(
        model=agent_config.model,
        temperature=agent_config.lookup_sales_data.temp_min,
        max_tokens=agent_config.lookup_sales_data.max_tokens,
        provider=agent_config.provider,
        ollama_url=agent_config.ollama_url,
        openai_api_key=agent_config.openai_api_key,
        schema=schema,
        agent_config=agent_config,
        enable_tracing=tracing.get('enabled', False),
        phoenix_endpoint=tracing.get('phoenix_endpoint'),
        phoenix_api_key=tracing.get('phoenix_api_key'),
        project_name=tracing.get('project_name', 'evaluating-agent'),
        parameter_provider=parameter_provider,
    )

    print(f"Provider: {agent_config.provider} | Model: {agent_config.model}")
    print(f"Prompt: {prompt}")
    print(f"Mode: lookup_only={run_params.get('lookup_only')}, no_vis={run_params.get('no_vis')}")
    if run_params.get('run_id'):
        print(f"Run ID: {run_params['run_id']}")
    print("-" * 60)

    raw_result = agent.run(prompt, **run_params)

    # run() returns a dict when using caching API, or (dict, score_variance) tuple
    # from the old best-of-n path
    if isinstance(raw_result, tuple):
        result, score_variance = raw_result
        print(f"Score variance: {score_variance:.4f}")
    else:
        result = raw_result

    artifact_dir = None
    if save_execution_artifacts:
        artifact_dir = _write_execution_artifacts(
            config_path=config_path,
            save_root=run_params.get("save_dir"),
            prompt=prompt,
            result=result,
            agent_config=agent_config,
            run_params=run_params,
        )

    # Output results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    if isinstance(result, dict):
        if "error" in result:
            print(f"Error: {result['error']}")
        if "answer" in result:
            for i, ans in enumerate(result["answer"]):
                print(f"\n--- Step {i+1} ---")
                print(ans[:2000])
        if "sql_query" in result:
            print(f"\nSQL: {result['sql_query']}")
        if "data" in result and result["data"]:
            preview = result["data"][:500]
            print(f"\nData preview:\n{preview}")
        if "run_id" in result:
            print(f"\nRun ID: {result['run_id']}")
        # Print GT tracking scores if available
        if "_gt_score" in result:
            print(f"\nGT tracking score: {result['_gt_score']:.3f}")
        if "_all_gt_scores" in result:
            print(f"All GT scores: {[f'{s:.3f}' for s in result['_all_gt_scores']]}")
    else:
        print(result)
    if artifact_dir is not None:
        print(f"\nSaved execution artifacts to: {artifact_dir}")

    # Show the plot if visualization code was generated
    if isinstance(result, dict) and result.get("chart_config"):
        answers = result.get("answer", [])
        chart_code = answers[-1] if len(answers) > 1 else None
        if chart_code and "plt" in chart_code:
            print("\nDisplaying chart...")
            import matplotlib
            for backend in ("TkAgg", "Qt5Agg", "GTK3Agg"):
                try:
                    matplotlib.use(backend)
                    import matplotlib.pyplot as plt
                    break
                except ImportError:
                    continue
            else:
                print("No interactive backend available. Install one of: python3-tkinter (system), PyQt5 (pip), or PyGObject (pip)")
                plt = None
            namespace = {
                "data_df": result.get("data_df"),
                "config": result.get("chart_config", {}),
            }
            if plt is not None:
                try:
                    exec(chart_code, namespace)  # noqa: S102
                except Exception as e:
                    print(f"Chart display error: {e}")


if __name__ == "__main__":
    main()
