## Sales Data Agent

An LLM-powered agent that queries a local parquet dataset with DuckDB, analyzes the results, and optionally generates visualization code. It uses a LangGraph workflow and supports **multiple LLM providers**:

- **OpenAI (default)**: e.g. `gpt-4o-mini`
- **Ollama (local)**: e.g. `llama3.2:3b`
- **Anthropic (Claude models)**: e.g. `anthropic:claude-3-5-sonnet-latest`

### What it does
- **Lookup**: converts natural language into SQL via the LLM and runs it on DuckDB over the parquet files in `data/`.
- **Analyze**: asks the LLM to summarize/interpret the results.
- **Visualize**: requests a chart configuration and generates matplotlib code to plot it.

Each step supports **best-of-N self-consistency**: the agent runs each step N times with a sampling schedule (temperature, top-p, or top-k) and selects the best candidate via an evaluator (consensus CSV IoU, LLM judge, or visualization judge).

---

## Requirements
- Python 3.10+
- An API key for OpenAI **or** Ollama running locally (`https://ollama.com`) with a model pulled
- Parquet file(s) present in `data/`

Install Python deps (from the project root):
```bash
pip install -r requirements.txt
```

---

## Project layout
```
DataAgent-1/
  Agent/
    data_agent.py          # SalesDataAgent class and LangGraph wiring
    config.py              # AgentConfig (per-step hyperparameters)
    steps.py               # Individual step implementations
    cache.py               # Run caching
    schema.py              # Multi-table schema support
    parameter_provider.py  # Interactive terminal parameter overrides
    tracing.py             # Phoenix/OpenTelemetry tracing helpers
    utils.py
  config/
    run_config.yaml        # Main single-run configuration file
    search_space.yaml      # (used by bulk runner)
    *.yaml                 # Additional config templates
  data/                    # Parquet files + per-table schema YAML files
  evaluation/
    bulk_runner.py         # Bulk ablation study runner
    run_benchmark.py       # Single benchmark run over a dataset
    benchmark_dataset.json # Benchmark prompts + ground truth
    search_space.yaml      # Hyperparameter search space for bulk runs
    aggregate_results.py   # Post-run aggregation utilities
  runs/                    # Output directory for run artifacts
  run_agent.py             # Entry point for single runs
  requirements.txt
```

---

## Running the agent

There are two ways to execute the agent: **single run** via `run_agent.py` or **bulk run** via `evaluation/bulk_runner.py`.

### Single run — `run_agent.py`

All parameters are configured in a YAML file (default: `config/run_config.yaml`). Edit the config, then run:

```bash
python run_agent.py                        # uses config/run_config.yaml
python run_agent.py config/my_config.yaml  # custom config path
```

On first launch the script shows the loaded config and optionally prompts for interactive overrides (set `interactive_config: true` in the YAML to enable).

#### Key run parameters (`run:` section in YAML)

| Parameter | Description |
|---|---|
| `prompt` | Natural language query |
| `visualization_goal` | Chart description (empty to skip) |
| `agent_mode` | `lookup_only` \| `analysis` \| `full` |
| `run_id` | Stable ID for caching / reproducibility (`null` = auto) |
| `save_dir` | Root output directory |
| `save_execution_artifacts` | Write `run_metadata.json` + `result.json` per run |
| `enable_codecarbon` | Enable CodeCarbon energy tracking |
| `reuse_from` | Run ID to reuse cached intermediate results from |
| `step_overrides` | Temporary per-step hyperparameter overrides |

#### Key agent parameters (`agent:` section in YAML)

| Parameter | Description |
|---|---|
| `model` | LLM model name (e.g. `gpt-4o-mini`, `llama3.2:3b`) |
| `provider` | `openai` or `ollama` |
| `ollama_url` | Ollama server URL (ignored for openai) |
| `openai_api_key` | `null` = read from `OPENAI_API_KEY` env var |

#### Per-step configuration (`steps:` section in YAML)

Each step (`decide_tool`, `lookup_sales_data`, `analyzing_data`, `create_visualization`) supports:

| Parameter | Description |
|---|---|
| `n` | Number of best-of-N candidates |
| `temp_min` / `temp_max` | Temperature range across candidates |
| `top_p_min` / `top_p_max` | Top-p range (alternative BoN axis) |
| `top_k_min` / `top_k_max` | Top-k range (Ollama only; alternative BoN axis) |
| `cot_n` | Chain-of-thought refinement iterations |
| `max_tokens` | Max tokens per generation |
| `use_cache` | Enable result caching |
| `eval` | Evaluator: `default` (consensus/LLM judge) or `none` |
| `enabled` | Enable/disable the step entirely |

#### Ground truth (optional)

Provide a ground-truth block in the YAML to log evaluation scores alongside each step (scores are never used to steer selection — only for tracking):

```yaml
ground_truth:
  csv_path: "path/to/gt_data.csv"
  analysis_text: "expected analysis text"
  vis_config: null
  vis_code: null
```

---

### Bulk run — `evaluation/bulk_runner.py`

The bulk runner performs a **3-phase ablation study** over the hyperparameter search space defined in `evaluation/search_space.yaml`. In each phase only one step's hyperparameters are varied across N randomly sampled configurations; the other steps are kept at their defaults. Results are aggregated automatically at the end of each phase.

```bash
# Validation run (1 config per phase, no think time)
python evaluation/bulk_runner.py \
    evaluation/benchmark_dataset.json \
    evaluation/search_space.yaml \
    --n-configs 1 --think-time 0 \
    --save-dir runs/bulk_results/validation

# Full 50+50+50 run
python evaluation/bulk_runner.py \
    evaluation/benchmark_dataset.json \
    evaluation/search_space.yaml \
    --n-configs 50 --think-time 5.0 \
    --save-dir runs/bulk_results/full_run

# Resume or run only one specific phase
python evaluation/bulk_runner.py ... --vary-step lookup_sales_data --resume
```

Results (per-config JSON + aggregated CSV/XLSX) are saved under `--save-dir`.

---

## LLM provider configuration

Set the provider and model in the `agent:` block of the YAML config. API keys can be passed directly or via environment variables:

```bash
# PowerShell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:ANTHROPIC_API_KEY="YOUR_KEY"
$env:OLLAMA_HOST="http://localhost:11434"
```

### Using Ollama locally

1. Install Ollama from `https://ollama.com/download`.
2. Pull a model: `ollama pull llama3.2:3b`
3. Start the server: `ollama serve`
4. Set `provider: "ollama"` and `model: "llama3.2:3b"` in the config YAML.

---

## Tracing with Phoenix (optional)

Enable OpenInference/Phoenix tracing to visualize agent runs. Configure in the `tracing:` block of the YAML:

```yaml
tracing:
  enabled: true
  phoenix_endpoint: "http://localhost:6006/v1/traces"
  phoenix_api_key: null   # required for Phoenix Cloud
  project_name: "evaluating-agent"
```

Install tracing dependencies:
```bash
pip install arize-phoenix openinference-instrumentation-langchain opentelemetry-api
```

Start Phoenix locally:
```bash
phoenix serve
```

Open the UI at `http://localhost:6006`. Top-level spans: `AgentRun`, `tool_choice`, `sql_query_exec`, `data_analysis`, `gen_visualization`.

---

## Energy and emissions (CodeCarbon)

Set `enable_codecarbon: true` in the `run:` block of the YAML config. Energy usage and CO₂ emissions are measured per-LLM-call and saved in `run_metadata.json` alongside each run's artifacts.

```bash
# View the Carbonboard dashboard (optional)
carbonboard --filepath "codecarbon/emissions.csv" --port 8050
```

---

## High-level flow

1. **Decide tool** (LLM): choose lookup → analyze → visualize → end.
2. **Lookup** (DuckDB): parquet → temp table → LLM SQL → query → text table in state.
3. **Analyze** (LLM): summarize / answer with reference to the result data.
4. **Visualize** (LLM): emit compact config → generate matplotlib code to plot.

Each step runs best-of-N candidates with a sampling schedule and selects the best via an evaluator. The agent exposes a single `run(prompt, ...)` entry point and returns the final state with an ordered `answer` list.
