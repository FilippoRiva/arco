## Sales Data Agent

An LLM-powered agent that queries a local parquet dataset with DuckDB, analyzes the results, and generates visualization code. It uses a LangGraph workflow and supports **multiple LLM providers**:

- **Ollama (local, default)**: `llama3.2:3b`
- **OpenAI (ChatGPT models)**: e.g. `openai:gpt-4o-mini`
- **Anthropic (Claude models)**: e.g. `anthropic:claude-3-5-sonnet-latest`

### What it does
- Lookup: converts natural language into SQL via the LLM and runs it on DuckDB over the parquet file at `data/Store_Sales_Price_Elasticity_Promotions_Data.parquet`.
- Analyze: asks the LLM to summarize/interpret the results.
- Visualize: requests a minimal chart configuration and emits matplotlib code to plot it.

---

## Requirements
- Python 3.10+
- Ollama running locally (`https://ollama.com`) with a model pulled (default: `llama3.2:3b`) **OR** an API key for OpenAI/Anthropic
- Parquet file present at `data/Store_Sales_Price_Elasticity_Promotions_Data.parquet`

Install Python deps (from the project root):
```powershell
pip install -r requirements.txt
```

---

## Start Ollama locally

1) Install Ollama for your OS from `https://ollama.com/download`.

2) Pull a model (default used in code is `llama3.2:3b`):
```powershell
ollama pull llama3.2:3b
```

3) Start the Ollama server (keeps API on `http://localhost:11434`):
```powershell
ollama serve
```

4) Verify it is running:
```powershell
curl http://localhost:11434/api/version
ollama list
ollama ps
```

If you need to run against a remote server, ensure the API is reachable and set any required environment variables or update the code to point to the remote host.

---

## Project layout
```
DataAgent/
  Agent/
    data_agent.py        # SalesDataAgent class and LangGraph wiring
  data/
    Store_Sales_Price_Elasticity_Promotions_Data.parquet
  LangChainAgent.ipynb   # Original notebook prototype
  readme.MD              # This file
```

---

## Using the agent

### From a Jupyter notebook
```python
from Agent.data_agent import SalesDataAgent

agent = SalesDataAgent(
    # Ollama (default): "llama3.2:3b"
    # OpenAI: "openai:gpt-4o-mini" (requires OPENAI_API_KEY)
    # Anthropic: "anthropic:claude-3-5-sonnet-latest" (requires ANTHROPIC_API_KEY)
    model="llama3.2:3b",
    temperature=0.1,
    max_tokens=2000,
    streaming=True,
)

result = agent.run(
    "Show me the sales in Nov 2021",
    visualization_goal="Sales trend for Nov 2021",
)

print("Final tool:", result.get("tool_choice"))
print("Chart config:", result.get("chart_config"))
print("Answer steps:", len(result.get("answer", [])))

# If the last answer is chart code, execute it to render the chart
if result.get("chart_config") and result.get("answer"):
    exec(result["answer"][-1], globals(), locals())
```

---

## Tracing with Phoenix (optional)

You can enable OpenInference/Phoenix tracing to visualize your agent runs. The top-level business spans remain `AgentRun`, `tool_choice`, `sql_query_exec`, `data_analysis`, and `gen_visualization`, with nested runtime spans for candidate execution, CoT refinement, cache decisions, SQL generation/execution, chart generation, and visualization validation.

### 1) Install tracing dependencies
```powershell
pip install arize-phoenix openinference-instrumentation-langchain opentelemetry-api
```

### 2) Choose where to send traces
- Self-hosted Phoenix (local): set endpoint to `http://localhost:6006/v1/traces`
- Phoenix Cloud: use endpoint `https://app.phoenix.arize.com/v1/traces` and your API key

### 3) Start Phoenix locally (self-hosted) (Only if running locally)
```powershell
phoenix serve
# or explicitly
# phoenix serve --host 0.0.0.0 --port 6006
```

Open the UI at `http://localhost:6006`.

### 4) Enable tracing in SalesDataAgent
```python
from Agent.data_agent import SalesDataAgent

# Self-hosted example
agent = SalesDataAgent(
    enable_tracing=True,
    phoenix_endpoint="http://localhost:6006/v1/traces",
    project_name="evaluating-agent",
)

# Cloud example
# agent = SalesDataAgent(
#     enable_tracing=True,
#     phoenix_endpoint="https://app.phoenix.arize.com/v1/traces",
#     phoenix_api_key="<YOUR_API_KEY>",
#     project_name="evaluating-agent",
# )

ret = agent.run("What was the most popular product SKU?")
```

Alternatively, set the endpoint via environment variable before creating the agent:
```python
import os
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006/v1/traces"
from Agent.data_agent import SalesDataAgent
agent = SalesDataAgent(enable_tracing=True)
```

### 5) View your traces
- Self-hosted UI: open `http://localhost:6006`, go to Traces, select the project (default: `evaluating-agent`).
- Cloud UI: open `https://app.phoenix.arize.com`, Traces → select your project.

You should see spans named: `AgentRun`, `tool_choice`, `sql_query_exec`, `data_analysis`, `gen_visualization`.
You should also see nested spans such as `cache_lookup`, `step_candidate`, `cot_refinement`, `sql_generation`, `sql_execution`, `chart_config_extraction`, `chart_code_generation`, `visualization_validation`, and `cache_save_run`.

### Troubleshooting tracing
- Verify the console shows: `[LangGraph] Starting LangGraph execution with tracing`.
- Confirm the endpoint includes `/v1/traces` and is reachable.
- Make sure dependencies are installed: `phoenix`, `openinference-instrumentation-langchain`, `opentelemetry-api`.
- For Cloud, ensure `phoenix_api_key` is set and valid.

### From the command line
```powershell
python -m Agent.data_agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

You can override the parquet path:
```powershell
python -m Agent.data_agent "Top products by revenue" --data "C:\path\to\your.parquet"
```

### Save the lookup result to CSV (no analysis, no visualization)
```powershell
python -m Agent.data_agent "What were the sales in November 2021?" --lookup-only --output-csv "results/sales_november_2021.csv"
```

### Best-of-N self-consistency for SQL generation (lookup only)
Runs the lookup \(N\) times with a temperature schedule and saves only the best attempt.
```powershell
python -m Agent.data_agent "What were the sales in November 2021?" --lookup-only `
  --best-of-n 5 --best-of-n-temp-min 0.0 --best-of-n-temp-max 0.6 `
  --output-csv "results/sales_november_2021.csv"
```

### Compare generated CSV vs a ground-truth CSV (IoU metrics)
This uses a lightweight Python comparator (column/row/data IoU).
```powershell
python -m Agent.data_agent "What were the sales in November 2021?" --lookup-only `
  --output-csv "results/sales_november_2021.csv" `
  --expected-csv "results/real_sales_november_2021.csv"
```
---

## Optional: Evaluate results with a C++ comparator

Build the comparator (requires CMake):
```powershell
cd my_cpp
cmake -S . -B build -G "Ninja"  # or "Visual Studio 17 2022" on Windows
cmake --build build --config Release
cd ..
```

Run the agent and compare the produced CSV with an expected CSV (C++ comparator):
```powershell
python -m Agent.data_agent "Weekly sales in 2021" `
  --lookup-only `
  --output-csv "results/weekly_sales_2021.csv" `
  --expected-csv "C:\path\to\expected.csv" `
  --evaluator-exe ".\my_cpp\build\resultcmp.exe" `
  --eval-keys "week,store_id"
```

PowerShell example with backtick continuations:
```powershell
python -m Agent.data_agent "Weekly sales in 2021" `
  --lookup-only `
  --model "llama3.2:3b" `
  --output-csv "results/weekly_sales_2021.csv" `
  --expected-csv "C:\path\to\expected.csv" `
  --evaluator-exe ".\my_cpp\build\resultcmp.exe" `
  --eval-keys "week,store_id"
```

The final returned dict will include an `evaluation` field like:
```json
{
  "equal": true,
  "row_count_actual": 1245,
  "row_count_expected": 1245,
  "mismatched_rows": 0,
  "mismatched_columns": [],
  "duration_ms": 37,
  "exit_code": 0
}
```

---

## Run with Docker

Build the image (from project root):
```powershell
docker build -t data-agent .
```

Build explicitly with this Dockerfile (if building from elsewhere):
```powershell
docker build -f Dockerfile -t data-agent .
```

Run (Docker Desktop on Windows/macOS):
```powershell
docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
  data-agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

Run (Linux host, replace with your host IP or network alias):
```bash
docker run --rm -e OLLAMA_HOST=http://192.168.1.10:11434 \
  data-agent "Top products by revenue"
```

Use a custom parquet by mounting it over the default path inside the container:
```powershell
docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v C:\path\to\your.parquet:/app/data/Store_Sales_Price_Elasticity_Promotions_Data.parquet \
  data-agent "Show weekly sales trend in 2021"
```

Override the entrypoint (run any Python you want inside the image):
```powershell
docker run --rm -it --entrypoint bash data-agent
# inside the container:
python -m Agent.data_agent "Top products by revenue" --goal "Top-5"
```

Notes:
- The container expects an Ollama server to be reachable at `OLLAMA_HOST`.
- Default `ENTRYPOINT` is `python -m Agent.data_agent`; any arguments after the image name are passed to the agent.

---

## Download and Install Apache JMeter

To perform load testing and performance analysis, you may want to use Apache JMeter. Follow these steps to download and install JMeter:

1. **Download JMeter:**
   - Visit the [Official Apache JMeter website](https://jmeter.apache.org/download_jmeter.cgi).
   - Download the binary archive for your operating system (e.g., `apache-jmeter-5.4.1.zip`).
   - Make sure to install java+8 [Java Download](https://www.java.com/en/download/manual.jsp)

2. **Extract the Archive:**
   - Extract the contents of the downloaded ZIP file to a directory of your choice. This directory will be your JMeter home. We created a folder and put in this project directly.

3. **Run JMeter:**
   - Navigate to the `bin` directory of your JMeter installation.
   - Execute the following command to start JMeter's GUI:
     - **Windows**:
       ```bash
       jmeter.bat
       ```
     - **macOS/Linux**:
       ```bash
       ./jmeter
       ```

4. **Verify the Installation:**
   - Once JMeter starts, you should see the JMeter interface.
   - You can now create test plans and perform load testing.

---

## Configuration
- Change model: pass `model="<name>"` to `SalesDataAgent(...)`.
- Custom parquet: pass `data_path="..."` in the constructor, or use `--data` in CLI.
- Visualization goal: pass `visualization_goal="..."` to `run()` or `--goal` in CLI.

### LLM provider configuration (models)
- **Ollama (default)**: `--model "llama3.2:3b"` (requires Ollama running; can override host with `OLLAMA_HOST`)
- **OpenAI**: `--model "openai:gpt-4o-mini"` (requires `OPENAI_API_KEY`)
- **Anthropic**: `--model "anthropic:claude-3-5-sonnet-latest"` (requires `ANTHROPIC_API_KEY`)

Environment variables (PowerShell):
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:ANTHROPIC_API_KEY="YOUR_KEY"
$env:OLLAMA_HOST="http://localhost:11434"
```

---

## Run as an HTTP API (Flask)
Start the server:
```powershell
python agent_api.py
```

Call the agent (example: lookup-only, best-of-n, save CSV, compare IoU):
```powershell
$body = @{
  prompt = "What were the sales in November 2021?"
  lookup_only = $true
  best_of_n = 5
  best_of_n_temp_min = 0.0
  best_of_n_temp_max = 0.6
  output_csv = "results/sales_november_2021.csv"
  expected_csv = "results/real_sales_november_2021.csv"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/call-agent" -ContentType "application/json" -Body $body
```

Use the C++ comparator via the API by adding:
- `evaluator_exe`: e.g. `".\my_cpp\build\resultcmp.exe"`
- `eval_keys`: e.g. `"week,store_id"`

---

## Optional: SPICE (Java) for analysis-text evaluation

SPICE is a **semantic scene-graph** metric originally designed for **image captions**. You *can* apply it to short, caption-like analysis outputs, but it may be noisy on long analytical text.

### Prerequisites
- **Java** installed (`java -version`)
- A SPICE jar file (e.g., `spice-1.0.jar`) downloaded from the official repo(https://panderson.me/spice/).

### CLI example (analyze-only + SPICE)
```powershell
python -m Agent.data_agent "What were the sum of sales in November 2021?" `
  --analyze-only `
  --expected-analysis-file "results/expected_analysis.txt" `
  --analysis-metric spice `
  --spice-jar "spice/spice-1.0.jar"
```

### API example (SPICE)
```json
{
  "prompt": "What were the sum of sales in November 2021?",
  "analyze_only": true,
  "expected_analysis": "…ground truth…",
  "analysis_metric": "spice",
  "spice_jar": "spice/spice-1.0.jar",
  "spice_java_bin": "java"
}
```

---

## Troubleshooting
- Check Ollama is up:
  - `curl http://localhost:11434/api/version`
  - `ollama list` and `ollama ps`
- Ensure the model is pulled (e.g., `ollama pull llama3.2:3b`).
- If you see connection errors, confirm no firewall is blocking `localhost:11434`.
- If SQL fails, skim the printed `data`/columns and adjust the prompt (e.g., include date formatting hints like `CAST(date_col AS VARCHAR)`).

---

## High-level flow
1. Decide tool (LLM): choose lookup → analyze → visualize → end.
2. Lookup (DuckDB): parquet → temp table → LLM SQL → query → text table in state.
3. Analyze (LLM): summarize/answer with reference to the result data.
4. Visualize (LLM): emit compact config → generate matplotlib code to plot.

The agent exposes a single `run(prompt, visualization_goal=None, initial_state=None)` entry point and returns the final state with an ordered `answer` list (analysis and then chart code when applicable).

---

## Energy and emissions (CodeCarbon)

This project integrates [CodeCarbon](https://mlco2.github.io/codecarbon/) to estimate energy usage and CO₂ emissions for each agent run.

- Enabled inside `SalesDataAgent.run(...)` via `EmissionsTracker`.
- Every execution writes a row to `codecarbon/emissions.csv` in the current working directory.

Install (already included if you use the project requirements):
```powershell
pip install -r requirements.txt
# or
pip install codecarbon
```

Run the agent as usual (CLI or API). After a run, inspect the log:
```powershell
Get-ChildItem codecarbon
type codecarbon\emissions.csv
```

Optional dashboard (Carbonboard):
```powershell
carbonboard --filepath "codecarbon/emissions.csv" --port 8050
```
Open `http://localhost:8050/`.

Notes:
- On Windows, CodeCarbon works without special drivers; it may use modeled power if sensors are unavailable.
- Logs are estimates; keep the machine plugged in and avoid heavy background tasks for more stable readings.


## Building agent_config_runner

### Quick Build

```bash
# Compile directly
g++ -std=c++11 -o agent_config_runner agent_config_runner.cpp

# Or with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Usage

```bash
# Use default agent_config.yaml in current directory
./agent_config_runner

# Specify custom config file
./agent_config_runner path/to/my_config.yaml
```

### Single Query Mode

Set in agent_config.yaml:
```yaml
prompt: "Your question"
run_batch: false
```

### Batch Mode

Set in agent_config.yaml:
```yaml
test_cases_json: ./test_cases.json
run_batch: true
```

### Features

✓ Efficient YAML parsing (no external dependencies)
✓ Batch processing with progress tracking
✓ Temporary file handling for gt_data/gt_analysis
✓ Per-test-case save directories
✓ Success/failure reporting
✓ Minimal overhead
