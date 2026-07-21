# Benchmark Config Reference

Configuration reference for `arco-cli benchmark` configs. A config file has three top-level sections: [`global`](#global) (required), [`defaults`](#defaults) (optional), and [`runs`](#runs) (required).

```yaml
global:
  schema: "..."
  # ...additional global options
defaults:
  Retriever:
    # ...per-agent default overrides
  Analyzer:
    # ...per-agent default overrides
runs:
  - name: "baseline"
    description: "..."
  - name: "high-temp-retriever"
    description: "..."
    changes:
      Retriever:
        temp_min: 1.2
```

---

## global

*Required.* Global execution parameters, provider credentials, caching rules, and runtime environmental toggles used by the agent infrastructure.

### Required fields

| Field | Type | Description |
|---|---|---|
| `schema` | string | Path to YAML schemas for the database, relative to this config file. |

### Optional fields

| Field | Type | Description |
|---|---|---|
| `run_id` | string | Unique identifier for this specific execution. Omit to auto-generate a unique ID, or set a fixed string to enable reproducible caching behavior. |
| `orchestration_enabled` | boolean | Determines the execution graph. If `true`, the workflow becomes LLM-orchestrated. |
| `empower` | boolean | Enables Arco empowerment. Defaults to `true`. |
| `enable_budget_controller` | boolean | Enables the Arco budget controller. Defaults to `true`. |
| `provider` | string (enum) | The AI model provider backend. One of: `openai`, `ollama`, `openrouter`. |
| `model` | string | The specific model identifier. |
| `ollama_url` | string (URI) | Base URL for local Ollama instances. Required if `provider` is `ollama`. |
| `save_state` | boolean | Whether final output artifacts should be persisted to disk. |
| `save_dir` | string | Directory path where output artifacts and execution metrics are saved. Also used for CodeCarbon results. Default: `./output`. |
| `enable_codecarbon` | boolean | Enables CodeCarbon emissions/consumption measurements. Output directory is set via `save_dir`. |
| `enable_tracing` | boolean | Enables Arize Phoenix tracing. |
| `phoenix_endpoint` | string (URI) | Endpoint of the Arize Phoenix client. |
| `phoenix_project_name` | string | Project name registered with the Arize Phoenix client. |

> **Note:** unlike earlier versions of this schema, `global` no longer carries `prompt` — the execution instruction and any output-plotting guidance are expected to live elsewhere (e.g. per-run) in this version.

---

## defaults

*Optional.* A dictionary of dynamically named agent definitions (e.g., `Retriever`, `Analyzer`) that establish the baseline configuration for each agent. Keys are arbitrary agent names chosen by the user; each value is an agent configuration object. All fields within an agent object are optional and, when set, override the corresponding `global` defaults for that agent only. Individual [`runs`](#runs) can further override these per-agent defaults via their `changes` block.

```yaml
defaults:
  Retriever:
    provider: openai
    model: gpt-4.1
    n: 3
    bon_param: temperature
    temp_min: 0.2
    temp_max: 0.9
    use_cache: true
    cache_mode: rw
    enabled: true
```

### Agent object fields

| Field | Type | Description |
|---|---|---|
| `provider` | string (enum) | The AI model provider backend for this agent. One of: `openai`, `ollama`. |
| `model` | string | The specific model identifier for this agent. |
| `n` | integer (≥ 1) | Number of completions to generate. |
| `bon_param` | string (enum) | Best-of-N selection parameter. One of: `temperature`, `top_k`, `top_p`. |
| `temp_min` | number (0.0–2.0) | Minimum sampling temperature limit. |
| `temp_max` | number (0.0–2.0) | Maximum sampling temperature limit. |
| `cot_n` | integer (≥ 1) | Chain-of-thought generation count multiplier (optional). |
| `top_p_min` | number (0.0–1.0) | Minimum top-p (nucleus sampling) threshold limit. |
| `top_p_max` | number (0.0–1.0) | Maximum top-p (nucleus sampling) threshold limit. |
| `top_k_min` | integer (≥ 1) | Minimum top-k token cutoff pool limit. |
| `top_k_max` | integer (≥ 1) | Maximum top-k token cutoff pool limit. |
| `max_tokens` | integer | Maximum number of tokens the agent should output per generation. |
| `enabled` | boolean | Master toggle to enable or disable this agent entirely. |
| `eval` | string (enum) | Evaluation system used by the agent to perform best-of-N selection. Currently: `default`. |

---

## runs

*Required.* An ordered array of benchmark runs. Each entry optionally applies a named set of overrides (`changes`) to specific agents defined under [`defaults`](#defaults). Runs without `changes` execute using the `global`/`defaults` configuration as-is.

### Run object fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Unique identifier for this benchmark run. |
| `description` | string | No | Human-readable explanation of what this benchmark tests. |
| `changes` | object | No | Per-agent overrides applied for this run. Keys are agent names matching entries under `defaults`. Omit entirely to run with defaults unchanged. |

### `changes` entry fields

Each key in `changes` is an agent name, and its value is an override object with the following optional fields (note: this object does **not** accept `additionalProperties` — only the fields below are allowed):

| Field | Type | Constraints |
|---|---|---|
| `provider` | string (enum) | `openai`, `ollama` |
| `model` | string | — |
| `n` | integer | ≥ 1 |
| `bon_param` | string (enum) | `temperature`, `top_k`, `top_p` |
| `temp_min` | number | 0.0–2.0 |
| `temp_max` | number | 0.0–2.0 |
| `cot_n` | integer | ≥ 1 |
| `top_p_min` | number | 0.0–1.0 |
| `top_p_max` | number | 0.0–1.0 |
| `top_k_min` | integer | ≥ 1 |
| `top_k_max` | integer | ≥ 1 |
| `max_tokens` | integer | ≥ 1 — maximum output tokens for this agent |
| `enabled` | boolean | — |
| `eval` | string (enum) | `default` |

```yaml
runs:
  - name: "baseline"
    description: "Default configuration, no overrides."
  - name: "retriever-broad-sampling"
    description: "Widen Retriever's temperature range for this run only."
    changes:
      Retriever:
        temp_min: 0.1
        temp_max: 1.8
        n: 5
```

---

## Notes

- `bon_param` selects which sampling dimension (`temperature`, `top_k`, or `top_p`) is varied across the `n` completions for best-of-N generation; pair it with the matching `*_min`/`*_max` bounds (e.g. `bon_param: temperature` with `temp_min`/`temp_max`).
- When `provider: ollama` is set (globally or per agent), `ollama_url` must be provided at the global level.
- Precedence for any given agent field is: `runs[].changes.<agent>` (highest) → `defaults.<agent>` → `global` (lowest).
- Unlike `defaults.<agent>` objects, `changes.<agent>` objects in `runs` explicitly disallow unknown properties (`additionalProperties: false`) — only the fields listed above are accepted.
- Every run must have a unique `name`; `changes` is optional and, when omitted, the run simply replays the `global`/`defaults` configuration.