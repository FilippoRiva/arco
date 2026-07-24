import types
from pathlib import Path

import pandas as pd
from core.config import AgentConfig, ArcoConfig
from workflow import SalesDataWorkflow, TracingHelper

from data import ColumnSchema, DatabaseSchema, TableSchema


class MockSpan:
    def __init__(self, name, kind=None, parent=None):
        self.name = name
        self.kind = kind
        self.parent = parent
        self.attributes = {}
        self.inputs = []
        self.outputs = []
        self.statuses = []
        self.exceptions = []

    def set_attributes(self, attributes):
        self.attributes.update(attributes)

    def set_input(self, value):
        self.inputs.append(value)

    def set_output(self, value):
        self.outputs.append(value)

    def set_status(self, *args):
        self.statuses.append(args)

    def record_exception(self, exc):
        self.exceptions.append(exc)


class MockSpanContext:
    def __init__(self, tracer, name, kind=None):
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.span = None

    def __enter__(self):
        parent = self.tracer.stack[-1] if self.tracer.stack else None
        self.span = MockSpan(self.name, kind=self.kind, parent=parent)
        self.tracer.spans.append(self.span)
        self.tracer.stack.append(self.span)
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.tracer.stack.pop()
        return False


class MockTracer:
    def __init__(self):
        self.spans = []
        self.stack = []

    def start_as_current_span(self, name, **kwargs):
        return MockSpanContext(self, name, kind=kwargs.get("openinference_span_kind"))


class FakeLLM:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def invoke(self, prompt):
        return types.SimpleNamespace(content="ok")


def make_agent(tracer=None):
    agent = SalesDataWorkflow.__new__(SalesDataWorkflow)
    agent.trace_helper = TracingHelper(tracer)
    agent.tracer = tracer
    agent.tracing_enabled = tracer is not None
    agent.current_run_step_results = {}
    agent.parameter_provider = types.SimpleNamespace(get_step_config=lambda step_name, config, state: config)
    agent.default_provider = "openai"
    agent.default_model = "test-model"
    agent.streaming = False
    agent.ollama_url = "http://localhost:11434"
    agent.openai_api_key = "test"
    agent.schema = None
    agent.model_is_reachable = True
    agent.llm = FakeLLM()
    agent._get_llm = lambda **kwargs: FakeLLM(temperature=kwargs.get("temperature", 0.1))
    agent.cache = types.SimpleNamespace(save_run=lambda **kwargs: None)
    agent.agent_config = types.SimpleNamespace(
        get_step_config=lambda name: AgentConfig(agent_name=name, use_cache=False),
        to_dict=dict,
    )
    agent.graph = types.SimpleNamespace(invoke=lambda state: {**state, "answer": ["graph-answer"]})
    return agent


def test_execute_step_without_tracing_keeps_behavior():
    agent = make_agent()
    config = AgentConfig(agent_name="analyzing_data", use_cache=False)

    class CustomThing:
        def __str__(self):
            return "custom-thing"

    def core_fn(state, llm, trace_helper=None):
        return {
            **state,
            "answer": ["analysis"],
            "chart_config": {"meta": CustomThing()},
        }

    result = agent._execute_step_with_config("analyzing_data", {"prompt": "hello", "answer": []}, core_fn, config)

    assert result["answer"] == ["analysis"]
    assert agent.trace_helper.enabled is False


def test_step_tracing_emits_business_and_candidate_spans():
    tracer = MockTracer()
    agent = make_agent(tracer)
    config = AgentConfig(agent_name="lookup_sales_data", use_cache=False)

    def core_fn(state, llm, trace_helper=None):
        return {
            **state,
            "data": "a\n1\n",
            "data_df": pd.DataFrame({"a": [1]}),
            "sql_query": "SELECT 1",
        }

    result = agent._execute_step_with_config("lookup_sales_data", {"prompt": "hello", "answer": []}, core_fn, config)

    span_names = [span.name for span in tracer.spans]
    assert "sql_query_exec" in span_names
    assert "step_candidate" in span_names
    assert result["sql_query"] == "SELECT 1"
    top_span = next(span for span in tracer.spans if span.name == "sql_query_exec")
    assert top_span.outputs
    assert top_span.outputs[-1]["dataframe"]["rows"] == 1


def test_best_of_n_and_cot_are_traced():
    tracer = MockTracer()
    agent = make_agent(tracer)
    config = AgentConfig(
        agent_name="analyzing_data",
        use_cache=False,
        n=2,
        temp_min=0.1,
        temp_max=0.2,
        cot_n=2,
        eval_fn=lambda result, state: result["_run_idx"],
    )
    call_counter = {"count": 0}

    def core_fn(state, llm, trace_helper=None):
        call_counter["count"] += 1
        return {
            **state,
            "answer": [f"analysis-{call_counter['count']}"],
        }

    result = agent._execute_step_with_config("analyzing_data", {"prompt": "hello", "answer": []}, core_fn, config)

    span_names = [span.name for span in tracer.spans]
    assert span_names.count("step_candidate") == 2
    assert span_names.count("cot_refinement") == 2
    assert result["_best_idx"] == 1
    step_span = next(span for span in tracer.spans if span.name == "data_analysis")
    assert step_span.attributes["selected_candidate_index"] == 1


def test_cache_hit_is_traced_and_skips_core_execution():
    tracer = MockTracer()
    agent = make_agent(tracer)
    config = AgentConfig(agent_name="analyzing_data", use_cache=True, cache_mode="skip")
    called = {"count": 0}

    def core_fn(state, llm, trace_helper=None):
        called["count"] += 1
        return {**state, "answer": ["fresh"]}

    state = {
        "prompt": "hello",
        "answer": [],
        "cached_step_results": {
            "analyzing_data": [{"answer": ["cached"]}],
        },
    }
    result = agent._execute_step_with_config("analyzing_data", state, core_fn, config)

    assert called["count"] == 0
    assert result["answer"] == ["cached"]
    cache_span = next(span for span in tracer.spans if span.name == "cache_lookup")
    assert cache_span.outputs[-1]["cache_hit"] is True


def test_run_core_traces_run_model_check_and_cache_save():
    tracer = MockTracer()
    agent = make_agent(tracer)
    agent.model_is_reachable = False
    agent.cache = types.SimpleNamespace(save_run=lambda **kwargs: None)
    agent._execute_step_with_config = lambda step_name, state, core_fn, config: {
        **state,
        "data": "a\n1\n",
        "data_df": pd.DataFrame({"a": [1]}),
        "sql_query": "SELECT 1",
    }

    result = agent._run("hello", lookup_only=True, save_results=True)

    span_names = [span.name for span in tracer.spans]
    assert "AgentRun_LookupOnly" in span_names
    assert "model_access_check" in span_names
    assert "cache_save_run" in span_names
    assert result["run_id"]


class ScriptedLLM:
    def __init__(self, temperature=0.1):
        self.temperature = temperature

    def invoke(self, prompt):
        if "workflow orchestrator managing a data analysis pipeline" in prompt:
            if "Answers generated so far: ['" in prompt or 'Answers generated so far: ["' in prompt:
                if "Last tool used: analyzing_data" in prompt:
                    return types.SimpleNamespace(content="create_visualization")
                if "Last tool used: create_visualization" in prompt:
                    return types.SimpleNamespace(content="end")
            if "Last tool used: lookup_sales_data" in prompt:
                return types.SimpleNamespace(content="analyzing_data")
            return types.SimpleNamespace(content="lookup_sales_data")
        if "expert SQL developer specializing in DuckDB queries" in prompt:
            return types.SimpleNamespace(
                content=(
                    "SELECT Sold_Date, SUM(Total_Sale_Value) AS Total_Sale_Value "
                    "FROM sales GROUP BY Sold_Date ORDER BY Sold_Date"
                )
            )
        if "professional data analyst providing insights" in prompt:
            return types.SimpleNamespace(
                content="Daily sales were $100 on 2021-11-01 and $150 on 2021-11-02, for a total of $250."
            )
        if "data visualization expert designing chart configurations" in prompt:
            return types.SimpleNamespace(
                content='{"chart_type":"line","x_axis":"Sold_Date","y_axis":"Total_Sale_Value","title":"Daily Sales"}'
            )
        if "Python data visualization developer creating matplotlib charts" in prompt:
            return types.SimpleNamespace(
                content=(
                    "import matplotlib.pyplot as plt\n"
                    "import pandas as pd\n"
                    "x_data = data_df[config['x_axis']]\n"
                    "y_data = data_df[config['y_axis']]\n"
                    "plt.figure(figsize=(8, 4))\n"
                    "plt.plot(x_data, y_data, marker='o')\n"
                    "plt.xlabel(config['x_axis'])\n"
                    "plt.ylabel(config['y_axis'])\n"
                    "plt.title(config['title'])\n"
                    "plt.xticks(rotation=45, ha='right')\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                )
            )
        return types.SimpleNamespace(content="ok")


def test_end_to_end_graph_run_with_tracing(tmp_path):
    tracer = MockTracer()
    df = pd.DataFrame(
        {
            "Sold_Date": ["2021-11-01", "2021-11-02"],
            "Total_Sale_Value": [100.0, 150.0],
        }
    )
    parquet_path = Path(tmp_path) / "sales.parquet"
    df.to_parquet(parquet_path, index=False)

    schema = DatabaseSchema(
        tables=[
            TableSchema(
                name="sales",
                description="Daily sales facts",
                file_path=str(parquet_path),
                columns=[
                    ColumnSchema(name="Sold_Date", description="Sales date", data_type="DATE"),
                    ColumnSchema(name="Total_Sale_Value", description="Revenue", data_type="FLOAT"),
                ],
            )
        ]
    )

    agent = SalesDataWorkflow.__new__(SalesDataWorkflow)
    agent.default_provider = "ollama"
    agent.default_model = "scripted-model"
    agent.streaming = False
    agent.ollama_url = "http://localhost:11434"
    agent.openai_api_key = None
    agent.schema = schema
    agent.tracer = tracer
    agent.tracing_enabled = True
    agent.trace_helper = TracingHelper(tracer)
    agent.parameter_provider = types.SimpleNamespace(get_step_config=lambda step_name, config, state: config)
    agent.cache = types.SimpleNamespace(save_run=lambda **kwargs: None)
    agent.current_run_step_results = {}
    agent.model_is_reachable = True
    agent.llm = ScriptedLLM()
    agent._get_llm = lambda **kwargs: ScriptedLLM(temperature=kwargs.get("temperature", 0.1))

    agent.agent_config = ArcoConfig(model="scripted-model", provider="ollama")
    for step_name in ["decide_tool", "lookup_sales_data", "analyzing_data", "create_visualization"]:
        cfg = agent.agent_config.get_agent_config(step_name)
        cfg.use_cache = False
        cfg.n = 1
        cfg.cot_n = 1
        agent.agent_config.set_agent_config(step_name, cfg)

    agent.graph = agent._build_graph()

    result = agent._run(
        "Show me daily sales for early November 2021 and visualize them",
        visualization_goal="Plot daily sales",
        save_results=True,
    )

    assert len(result["answer"]) == 2
    assert result["chart_config"]["chart_type"] == "line"
    assert "Total_Sale_Value" in result["data"]
    assert "plt.plot" in result["answer"][-1]

    span_names = [span.name for span in tracer.spans]
    assert "AgentRun" in span_names
    assert "tool_choice" in span_names
    assert "sql_query_exec" in span_names
    assert "data_analysis" in span_names
    assert "gen_visualization" in span_names
    assert "sql_generation" in span_names
    assert "sql_execution" in span_names
    assert "chart_config_extraction" in span_names
    assert "chart_code_generation" in span_names
    assert "visualization_validation" in span_names
    assert "cache_save_run" in span_names


def test_step_config_serialization_preserves_llm_overrides():
    config = AgentConfig(
        agent_name="decide_tool",
        provider="openai",
        model="gpt-4o-mini",
        ollama_url="http://remote-ollama:11434",
    )

    serialized = config.to_dict()
    restored = AgentConfig.from_dict(serialized)

    assert restored.default_provider == "openai"
    assert restored.default_model == "gpt-4o-mini"
    assert restored.ollama_url == "http://remote-ollama:11434"


def test_agent_config_from_yaml_loads_step_level_llm_overrides(tmp_path):
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        """
agent:
  model: "gpt-4o-mini"
  provider: "openai"
  ollama_url: "http://localhost:11434"

steps:
  decide_tool:
    provider: "openai"
    model: "gpt-4.1-mini"
  lookup_sales_data:
    provider: "ollama"
    model: "llama3.2:3b"
    ollama_url: "http://ollama.internal:11434"

run:
  prompt: "test"
""",
        encoding="utf-8",
    )

    agent_config, run_params, schema = ArcoConfig.from_data_dir(str(config_path))

    assert schema is None
    assert run_params["prompt"] == "test"
    assert agent_config.orchestrator_config.default_provider == "openai"
    assert agent_config.orchestrator_config.default_model == "gpt-4.1-mini"
    assert agent_config.retriever_config.default_provider == "ollama"
    assert agent_config.retriever_config.default_model == "llama3.2:3b"
    assert agent_config.retriever_config.ollama_url == "http://ollama.internal:11434"


def test_execute_step_uses_step_specific_llm_override():
    agent = make_agent()
    created_llms = []

    def record_create_llm(**kwargs):
        created_llms.append(kwargs)
        return FakeLLM(temperature=kwargs.get("temperature", 0.1))

    agent._get_llm = record_create_llm
    agent.agent_config = ArcoConfig(model="global-model", provider="openai")
    config = AgentConfig(
        agent_name="decide_tool",
        use_cache=False,
        provider="ollama",
        model="llama3.2:3b",
        ollama_url="http://ollama-step:11434",
    )

    result = agent._execute_step_with_config(
        "decide_tool",
        {"prompt": "hello", "answer": []},
        lambda state, llm, trace_helper=None: {**state, "tool_choice": "lookup_sales_data"},
        config,
    )

    assert result["tool_choice"] == "lookup_sales_data"
    assert created_llms[0]["provider"] == "ollama"
    assert created_llms[0]["model"] == "llama3.2:3b"
    assert created_llms[0]["ollama_url"] == "http://ollama-step:11434"
    assert len(created_llms[0]["callbacks"]) == 1


def test_execute_step_inherits_global_llm_config_when_step_override_is_missing():
    agent = make_agent()
    created_llms = []

    def record_create_llm(**kwargs):
        created_llms.append(kwargs)
        return FakeLLM(temperature=kwargs.get("temperature", 0.1))

    agent._get_llm = record_create_llm
    agent.agent_config = ArcoConfig(model="gpt-4o-mini", provider="openai")
    config = AgentConfig(agent_name="analyzing_data", use_cache=False)

    result = agent._execute_step_with_config(
        "analyzing_data",
        {"prompt": "hello", "answer": []},
        lambda state, llm, trace_helper=None: {**state, "answer": ["analysis"]},
        config,
    )

    assert result["answer"] == ["analysis"]
    assert created_llms[0]["provider"] == "openai"
    assert created_llms[0]["model"] == "gpt-4o-mini"
    assert created_llms[0]["ollama_url"] == "http://localhost:11434"
    assert len(created_llms[0]["callbacks"]) == 1


def test_execute_step_supports_mixed_provider_runs():
    agent = make_agent()
    created_llms = []

    def record_create_llm(**kwargs):
        created_llms.append(kwargs)
        return FakeLLM(temperature=kwargs.get("temperature", 0.1))

    agent._get_llm = record_create_llm
    agent.agent_config = ArcoConfig(model="gpt-4o-mini", provider="openai")

    decide_config = AgentConfig(
        agent_name="decide_tool",
        use_cache=False,
        provider="openai",
        model="gpt-4.1-mini",
    )
    lookup_config = AgentConfig(
        agent_name="lookup_sales_data",
        use_cache=False,
        provider="ollama",
        model="llama3.2:3b",
        ollama_url="http://mixed-ollama:11434",
    )

    agent._execute_step_with_config(
        "decide_tool",
        {"prompt": "hello", "answer": []},
        lambda state, llm, trace_helper=None: {**state, "tool_choice": "lookup_sales_data"},
        decide_config,
    )
    agent._execute_step_with_config(
        "lookup_sales_data",
        {"prompt": "hello", "answer": []},
        lambda state, llm, trace_helper=None: {**state, "data": "a\n1\n", "sql_query": "SELECT 1"},
        lookup_config,
    )

    assert created_llms[0]["provider"] == "openai"
    assert created_llms[0]["model"] == "gpt-4.1-mini"
    assert len(created_llms[0]["callbacks"]) == 1
    assert created_llms[1]["provider"] == "ollama"
    assert created_llms[1]["model"] == "llama3.2:3b"
    assert created_llms[1]["ollama_url"] == "http://mixed-ollama:11434"
    assert len(created_llms[1]["callbacks"]) == 1


def test_create_llm_uses_defaults_and_attaches_callbacks(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(kind="openai", kwargs=kwargs)

    monkeypatch.setattr("arco.data_agent.ChatOpenAI", fake_chat_openai)
    agent = make_agent()
    cb = object()

    llm = SalesDataWorkflow._get_llm(
        agent,
        temperature=0.2,
        max_tokens=321,
        top_p=0.9,
        callbacks=[cb],
    )

    assert llm.kind == "openai"
    assert captured["model"] == "test-model"
    assert captured["callbacks"] == [cb]
    assert captured["api_key"] == "test"
    assert captured["streaming"] is False


def test_create_llm_uses_overrides_and_attaches_callbacks(monkeypatch):
    captured = {}

    def fake_chat_ollama(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(kind="ollama", kwargs=kwargs)

    monkeypatch.setattr("arco.data_agent.ChatOllama", fake_chat_ollama)
    agent = make_agent()
    cb = object()

    llm = SalesDataWorkflow._get_llm(
        agent,
        temperature=0.4,
        max_tokens=512,
        top_p=0.8,
        top_k=20,
        callbacks=[cb],
        provider="ollama",
        model="llama3.2:3b",
        ollama_url="http://override-ollama:11434",
    )

    assert llm.kind == "ollama"
    assert captured["model"] == "llama3.2:3b"
    assert captured["base_url"] == "http://override-ollama:11434"
    assert captured["callbacks"] == [cb]
    assert captured["top_k"] == 20
