# ARCO framework

An agentic workflow profiling framework compatible with any **LangGraph** workflow built using our `Agent` and `Evaluator` abstraction.

Compatible with **OpenAI**, **OpenRouter** and **Ollama** backends.

It provides:
- Single agent **Best-of-N** support 
- **Chain of Thought** integration
- Local **Energy and Emissions** profiling through codecarbon
- **Performance** profiling through a proper benchmarking interface

---

### System Requirements

Depending on the agents and models you use, you may also need:

- An available LLM backend:
  - OpenAI API access (for OpenAI-based agents)
  - Ollama installed and running locally (for local models)
- A compatible environment for profiling:
  - CodeCarbon supports CPU/GPU/RAM energy tracking
  - GPU monitoring requires compatible hardware and drivers

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/FilippoRiva/arco
cd arco
```
### 2. Create and activate a virtual environment

Using venv:

```bash
python -m venv .venv
```

Activate it:

#### - Linux/macOS

```bash
source .venv/bin/activate
```

#### - Windows

```bash
.venv\Scripts\activate
```

### 3. Install ARCO

For a standard installation use:

```bash
pip install .
```

For a development installation [optional] (providing testing and notebooks support) use:

```bash
pip install -e ".[dev]"
```

the `-e` flag allows for code changes to be immediately effective in the `arco-cli` command.

### 4. Verify Installation

After installation, verify that the CLI is available:

```bash
arco-cli --help
```

You should see the available ARCO commands.

### 5. Provider setup 

#### Ollama Setup [Optional]

If you plan to use local models (which is needed for a proper profiling of your agents), install Ollama and make sure the service is running:

```bash
systemctl status ollama
```

Then pull the desired model:

```bash
ollama pull <model-name>
```

#### OpenAI Setup [Optional]

For OpenAI-based agents, export the `OPENAI_API_KEY` environment variable containing your API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

#### OpenRouter Setup [Optional]

For OpenRouter-based agents, export the `OPENROUTER_API_KEY` environment variable containing your API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

---
## Usage

The entire functionality of this framework is exposed through the `arco-cli` command-line tool.

ARCO provides three main sub-commands:

- `arco-cli run` - execute a single agent workflow
- `arco-cli benchmark` - evaluate multiple configurations against a benchmark dataset
- `arco-cli cache` - inspect and manage previous executions

---

### `arco-cli run`

Executes a single ARCO workflow using a provided configuration file.

```bash
arco-cli run --config <path-to-config.yaml>
```

Options
```
--config	-c	Path to the ARCO configuration YAML file (required)
--verbose	-v	Display additional execution information, including agent configuration and metrics
```
Example
```bash
arco-cli run -c configs/example.yaml -v
```

Refer to [Run Configuration Files](docs/run_config.md) for writing run configuration files.

### `arco-cli benchmark`

Runs a benchmark suite by executing multiple ARCO configurations against a ground-truth dataset.

```bash
arco-cli benchmark \
    --dataset <path-to-dataset.json> \
    --config <path-to-benchmark.yaml>
```

Options
```
--dataset	-d	Path to the benchmark ground-truth dataset (required)
--config	-c	Path to the benchmark configuration YAML file (required)
--save-dir		Directory where benchmark results are stored (default: ./output/benchmarks)
--id		        Custom identifier for the benchmark run
--verbose	-v	Enable detailed visualization of agent executions
```

Example
```bash
arco-cli benchmark \
    -d datasets/sales_gt.json \
    -c benchmarks/config.yaml \
    --save-dir output/results
```

Benchmark results are automatically saved as CSV files containing execution metrics, evaluations, and profiling information.

Refer to [Benchmark Configuration Files](docs/benchmark_config.md) for writing benchmark configuration files.

### `arco-cli cache`

Provides tools to inspect, visualize, and manage cached ARCO executions.

Usage
```bash
arco-cli cache [options]
```
Options

```
--save-dir	-d	Directory containing cached runs (default: output)
--runs	    -r  List available cached executions
--stats	    -s	Display cache statistics
--view-run	-v	Visualize a specific cached run by ID
--delete		Delete a cached run by ID
--clear		    Remove all cached runs
```
Examples

List available runs:

```bash
arco-cli cache --runs
```

View a previous execution:

```bash
arco-cli cache --view-run <run_id>
```

Clear the cache:

```bash
arco-cli cache --clear
```

## Tracing with Phoenix [optional]

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

## Energy and emissions [CodeCarbon]

Set `enable_codecarbon: true` in the `run:` block of the YAML config. Energy usage and CO₂ emissions are measured per-LLM-call and saved in `run_metadata.json` alongside each run's artifacts.

```bash
# View the Carbonboard dashboard (optional)
carbonboard --filepath "codecarbon/emissions.csv" --port 8050
```

---