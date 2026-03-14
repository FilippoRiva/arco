#!/usr/bin/env python3
"""Run the DataAgent from a YAML configuration file.

Usage:
    python run_agent.py                           # uses config/run_config.yaml
    python run_agent.py config/my_config.yaml     # custom config path
"""
import sys
import os

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
