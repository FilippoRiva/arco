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


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/run_config.yaml"
    if not os.path.isfile(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")
    agent_config, run_params, schema = AgentConfig.from_yaml(config_path)

    # Extract tracing config (goes to SalesDataAgent.__init__, not run())
    tracing = run_params.pop('tracing', {})

    # Extract prompt (positional arg to run())
    prompt = run_params.pop('prompt')
    if not prompt:
        print("Error: 'run.prompt' is required in the YAML config")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
