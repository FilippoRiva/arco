# Experiment Catalog

ARCO can connect benchmark generation, execution, and analysis through named
experiments in `config/catalog.yaml`.

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
  sales-planned-gpt41-nano:
    description: Planned sales workflow evaluated with GPT-4.1 nano.
    workflow: planned_sales
    generation:
      config: config/run_config/sales/benchmarks_generation/planned.yaml
      prompts: data/prompts/sales/prompts.json
      output: data/benchmarks/sales/planned_ground_truth.json
    benchmark:
      config: config/benchmark_config/sales/gpt_4.1_nano_planned.yaml
      output_dir: output/benchmarks/sales-planned-gpt41-nano
```

Run the complete lifecycle using only the experiment ID:

```bash
arco generate-benchmark --experiment sales-planned-gpt41-nano
arco benchmark --experiment sales-planned-gpt41-nano
arco analyze-benchmark --experiment sales-planned-gpt41-nano
```

The explicit path-based interfaces remain available. For example:

```bash
arco generate-benchmark \
  --config config/run_config/sales/benchmarks_generation/planned.yaml \
  --prompts data/prompts/sales/prompts.json \
  --output data/benchmarks/sales/planned_ground_truth.json
```

When a benchmark is run through the catalog, its `bench_metadata.json` stores
the experiment ID and resolved catalog paths. This preserves the relationship
between the result and the exact experiment definition used to create it.
