# Run Configuration

Configuration reference for `arco-cli run`, the schema used to configure the Arco agent system. A config file has two top-level sections: [`global`](#global) (required) and [`agents`](#agents) (optional).

```yaml
global:
  prompt: "..."
  schema: "..."
  # ...additional global options
agents:
  Retriever:
    # ...per-agent overrides
  Analyzer:
    # ...per-agent overrides
```

---

## global

*Required.* Global execution parameters, provider credentials, caching rules, and runtime environmental toggles used by the agent infrastructure.

### Required fields

| Field | Type | Description |
|---|---|---|
| `prompt` | string | The execution instruction or query passed to the agent system. |
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

---

## agents

*Optional.* A dictionary of dynamically named agent definitions (e.g., `Retriever`, `Analyzer`). Keys are arbitrary agent names chosen by the user; each value is an agent configuration object. All fields within an agent object are optional and, when set, override the corresponding `global` defaults for that agent only.

```yaml
agents:
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

## Notes

- `bon_param` selects which sampling dimension (`temperature`, `top_k`, or `top_p`) is varied across the `n` completions for best-of-N generation; pair it with the matching `*_min`/`*_max` bounds (e.g. `bon_param: temperature` with `temp_min`/`temp_max`).
- When `provider: ollama` is set (globally or per agent), `ollama_url` must be provided at the global level.
- Per-agent settings override `global` settings only for that agent; any field left unset falls back to the global value.
- A `schema.json` file is provided in the `./config/run_config/`  folder that can be used by a YAML language server to check configuration correctness