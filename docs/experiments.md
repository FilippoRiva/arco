# Experiment Catalog

ARCO can connect benchmark generation, execution, and analysis through named
experiments in `config/catalog.yaml`. Module-specific configuration and
benchmark inputs live under `config/<module>/run/` and `config/<module>/bench/`;
Retriever source data remains outside `config/` under `data/datasets/`.

An experiment defines:

- the workflow;
- the run configuration used to generate ground truth;
- the prompt set;
- the generated ground-truth dataset location;
- the benchmark configuration;
- the benchmark output directory.

Example:

```yaml
experiments:
  demo-planned-gpt41-nano:
    description: Small planned sales benchmark used for demonstrations.
    workflow: planned_sales
    generation:
      config: config/sales/run/benchmarks_generation/planned.yaml
      prompts: config/sales/bench/prompts/demo_prompts.json
      output: config/sales/bench/data/demo/planned_ground_truth.json
    benchmark:
      config: config/sales/bench/demo_benchmark_config.yaml
      output_dir: output/benchmarks/demo-planned-gpt41-nano
```

Run the complete lifecycle using only the experiment ID:

```bash
arco generate-benchmark --experiment demo-planned-gpt41-nano
arco benchmark --experiment demo-planned-gpt41-nano
arco analyze-benchmark --experiment demo-planned-gpt41-nano
```

The explicit path-based interfaces remain available. For example:

```bash
arco generate-benchmark \
  --config config/sales/run/benchmarks_generation/planned.yaml \
  --prompts config/sales/bench/prompts/demo_prompts.json \
  --output config/sales/bench/data/demo/planned_ground_truth.json
```

When a benchmark is run through the catalog, its `bench_metadata.json` stores
the experiment ID and resolved catalog paths. This preserves the relationship
between the result and the exact experiment definition used to create it.

Before execution, ARCO copies the benchmark configuration and dataset to
`benchmark_config.yaml` and `dataset.json` in the experiment output directory,
then loads and executes from those snapshots. On resume, existing snapshots are
reused; if a supplied source file has changed, ARCO asks you to use a new
benchmark name or clear the old output rather than replacing the inputs behind
a checkpoint.

Each per-run CSV contains one row per benchmark entry. Along with `entry_id`,
`run_id`, and `run_fingerprint` (the effective configuration and ordered
dataset identity), it stores:

- `changes`: the declared run-specific configuration overrides as JSON;
- `state`: the complete final workflow state serialized as JSON. Analysis
  derives per-agent metrics and the ground-truth comparison trace from it.

For an experiment with `output_dir: output/benchmarks/my-experiment`, the flat
run CSVs are stored in `output/benchmarks/my-experiment/runs/`.

Each successfully evaluated entry is appended and flushed to the run CSV before
the next entry begins. If a run is interrupted, the next invocation restores
matching completed entries from that CSV and evaluates only the missing ones.
The benchmark command writes raw run CSVs and metadata, but does not compute a
`summary.csv`; aggregate analysis belongs to `analyze-benchmark`. The CLI reports
whether it is starting fresh, resuming from cached entries, or restoring a
fully completed run. A checkpoint created from a different
configuration or dataset is rejected rather than silently reused.
