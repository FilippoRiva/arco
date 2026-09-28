# Developing Agents, Evaluators, and Workflows

This guide explains how to extend ARCO with custom agents, evaluation logic, and
workflow graphs. 

## The basic idea

ARCO separates three concerns:

1. **Agents** do one unit of work. They receive the current immutable workflow
   state and an LLM client, then return a new state containing their answer.
2. **Evaluators** score candidate answers or compare an answer with reference
   data. They can also define which answer fields become benchmark ground truth.
3. **Workflows** connect agent instances in a graph. They define which agent
   runs first, what can run next, and when execution ends.

The workflow graph passes one `State` from node to node:

```text
Config + prompt
      │
      ▼
Workflow graph ──► Agent A ──► Agent B ──► ... ──► final State
                     │            │
                     └─ Answer    └─ Answer
                          │
                          └── Evaluator and profiling data
```

`State` carries the original prompt, run ID, per-agent configuration, and the
ordered list of `Answer` objects. Each answer contains a human-readable message,
structured output, optional evaluations, and profiling data. The state is
immutable: agents return a replacement state rather than mutating it in place.

## 1. Write an agent

Subclass `arco.core.Agent` and implement `core(state, llm)`. The framework
handles configuration lookup, LLM construction, best-of-N execution,
evaluation, timing, optional CodeCarbon tracking, and attaching profiling data.

```python
from typing import TYPE_CHECKING

from arco.core import Agent, AnswerDraft, Evaluator
from my_project.evaluators import AnswerEvaluator

if TYPE_CHECKING:
    from arco.core import LLM, State


class AnswerAgent(Agent):
    """Generate a concise answer to the workflow prompt."""

    @property
    def evaluator(self) -> Evaluator:
        return AnswerEvaluator()

    def core(self, state: State, llm: LLM) -> AnswerDraft:
        response = llm.invoke(
            "Answer the user's question in a JSON object with an `answer` field.\n"
            f"Question: {state.prompt}"
        )
        output = response.extract_json()
        message = str(output.get("answer", response.text))

        return AnswerDraft(message=message, output=output)
```

### Agent implementation notes

- `core` receives an `LLM` already configured from the workflow's run config.
  Use `llm.invoke(...)`; its response wrapper exposes `.text`,
  `.extract_json()`, `.extract_json_list()`, `.extract_sql()`, `.logprobs`, and
  `.reasoning`.
- Return an `AnswerDraft` containing the answer message, structured output,
  and optional error. The framework converts it into a complete `Answer`,
  attaching the agent ID, configuration, LLM metadata, token usage, and profiling.
- Put downstream-consumable structured values in `output`; use `message` for a
  readable summary. The next agent can access prior answers with
  `state.get_last_answer()` or `state.get_last_answer(AgentType("AnswerAgent"))`.
- Use a unique agent name in each workflow. By default, the class name becomes
  its ID; pass `agent_name="..."` to `super().__init__()` when multiple
  instances of a class need distinct IDs. The ID is used in graph nodes,
  answer records, evaluator lookup, and per-agent configuration.
- Raise `AgentException` when a required prior answer or dependency is missing.
  This makes the failure explicit instead of silently proceeding with invalid
  state.
- Keep `core` focused on one responsibility. Put reusable tools and domain
  logic in separate functions or modules.

Agents are not discovered through a separate global agent registry. A workflow
creates agent instances and registers them with `graph.add_agent(...)`.

## 2. Write an evaluator

Subclass `arco.core.Evaluator`. It has three evaluation hooks:

- `_eval(state, judge_provider, judge_model, llm_accumulator=None)` scores one
  candidate state for Best-of-N selection.
- `_batch_eval(states)` may score several candidate states together. Return a
  list of `Evaluation` objects in the same order, or return `None` to let ARCO
  call `_eval` for each candidate.
- `_gt_eval(answer, gt_data, judge_provider, judge_model)` compares an answer
  with the reference data for its position in a benchmark trace.

A deterministic exact-match evaluator might look like this:

```python
from arco.core import Answer, Evaluation, Evaluator, State


class AnswerEvaluator(Evaluator):
    def _eval(
        self,
        state: State,
        judge_provider: str,
        judge_model: str,
        llm_accumulator=None,
    ) -> Evaluation:
        answer = state.get_last_answer()
        has_answer = answer is not None and bool(answer.agent_output.get("answer"))
        return Evaluation(score=1.0 if has_answer else 0.0)

    def _batch_eval(self, states: list[State]) -> list[Evaluation] | None:
        # ARCO falls back to _eval once per candidate.
        return None

    def _gt_eval(
        self,
        answer: Answer,
        gt_data: dict,
        judge_provider: str,
        judge_model: str,
    ) -> Evaluation:
        matches = answer.agent_output.get("answer") == gt_data.get("answer")
        return Evaluation(score=1.0 if matches else 0.0)

    def extract_gt_from_answer(self, answer: Answer) -> dict:
        """Select the structured fields to save as this agent's reference."""
        return {"answer": answer.agent_output.get("answer")}
```

`Evaluation.score` is a normalized score in `[0, 1]`; `success` defaults to
`True`. A custom evaluator can use deterministic rules, external metrics, or an
LLM judge. `judge_provider` and `judge_model` are provided for evaluators that
need a judge; ARCO's evaluator implementations use the common LLM layer when
appropriate.

### How evaluators are used

- If an agent has an evaluator and its configured `n` is greater than one, ARCO
  uses the evaluator to rank the Best-of-N candidates. If `_batch_eval` returns
  `None`, ARCO evaluates candidates individually with `_eval`.
- During benchmark evaluation, ARCO walks the actual answers and expected trace
  in order. It evaluates an answer only when its agent ID matches the expected
  agent at that position. Once the agent sequence diverges, later answers are
  not ground-truth-evaluated.
- During benchmark generation, ARCO calls `extract_gt_from_answer` for each
  answer that has an evaluator. The returned dictionary is stored as that
  trace element's `data`. Keep its schema consistent with what `_gt_eval`
  expects. Without an evaluator, the generated trace element's `data` is `{}`.

An evaluator's exact-match test is only an example. For free-form language,
consider semantic scoring, structured validation, tolerances for numeric
values, or domain-specific metrics instead of literal string equality.

## 3. Build a workflow graph

Subclass `arco.core.Workflow`, assign a stable `workflow_id`, and implement
`initialize(config, graph)`. The supplied `arco.core.Graph` is a LangGraph
`StateGraph` specialized for ARCO's `State`; its helpers accept agent instances
directly.

A simple two-agent chain:

```python
from arco.core import Config, Workflow
from arco.core.graph import END
from arco.core import Graph

from my_project.agents import AnswerAgent, ReviewAgent


class AnswerReviewWorkflow(Workflow):
    workflow_id = "answer_review"
    description = "Generates an answer and then reviews it."

    def initialize(self, config: Config, graph: Graph) -> None:
        answer = AnswerAgent()
        review = ReviewAgent()

        graph.add_agent(answer)
        graph.add_agent(review)
        graph.set_entry_agent(answer)
        graph.add_agent_edge(answer, review)
        graph.add_agent_edge(review, END)
```

For conditional routing, add all possible agents to the graph, then provide a
routing function and route map:

```python
from arco.core import AgentType, Config, Graph, State, Workflow
from arco.core.graph import END


class RoutedWorkflow(Workflow):
    workflow_id = "routed_workflow"

    def initialize(self, config: Config, graph: Graph) -> None:
        router = RouterAgent()
        specialist = SpecialistAgent()
        fallback = FallbackAgent()

        for agent in (router, specialist, fallback):
            graph.add_agent(agent)
        graph.set_entry_agent(router)

        def choose_next(state: State) -> str:
            answer = state.get_last_answer(AgentType(router.name))
            if answer is None:
                return "fallback"
            return answer.agent_output.get("route", "fallback")

        graph.add_conditional_edges(
            source=router,
            path=choose_next,
            path_map={
                "specialist": specialist,
                "fallback": fallback,
                "end": END,
            },
        )
        graph.add_agent_edge(specialist, END)
        graph.add_agent_edge(fallback, END)
```

A route function returns a key represented in `path_map`. `Graph` also supports
ordinary edges and loops, such as sending a specialist back to a planner or
router. Keep route keys and the agent outputs that select them consistent.

### Workflow lifecycle and registration

A concrete `Workflow` subclass is added to `WorkflowFactory` when its Python
module is imported. `WorkflowFactory.get(config=...)` selects a class using
`config.workflow`. `Workflow.__init__` builds and compiles the graph by calling
`initialize`.

For an application or script that imports a custom workflow directly:

```python
import my_project.workflows.answer_review  # triggers workflow registration

from arco.core import Config, WorkflowFactory

config = Config(workflow="answer_review", prompt="What is ARCO?")
workflow = WorkflowFactory.get(config=config)
```

ARCO's built-in `load_workflows()` imports its built-in workflow modules. For a
workflow to be selectable through the stock `arco run` CLI, its module must be
imported before the workflow list/config is resolved. In a source checkout,
one simple option is to add the module name to the built-in module list in
`src/arco/workflows/__init__.py`. An external package can instead arrange for
its workflow module to be imported during application startup or provide its
own CLI entry point.

## 4. Configure and run a custom workflow

A run YAML names the registered workflow under `global.workflow`; global LLM
settings can be overridden for individual agent IDs under the top-level
`agents` mapping:

```yaml
global:
  workflow: answer_review
  prompt: "Explain why evaluation matters."
  default_provider: openai
  default_model: gpt-4.1-mini
  enable_codecarbon: false

agents:
  AnswerAgent:
    n: 1
  ReviewAgent:
    n: 1
```

After the workflow module is importable in the process, run it with:

```bash
arco run --config config/my_project/answer_review.yaml
```

Use the same agent IDs in the workflow, config, and any benchmark config. The
agent ID also determines which evaluator is associated with an answer and which
trace element it can match during benchmark evaluation. See
[Run Configuration](run_config.md) for all run fields and
[Benchmark Configuration](benchmark_config.md) for benchmark defaults and
per-run overrides.

## 5. Test and debug extensions

A useful development sequence is:

1. Unit-test `Agent.core` with a small fake LLM response; check that it returns
   a new state with the expected `Answer` and structured output.
2. Unit-test evaluator rules independently, including malformed or missing
   outputs, exact-match boundaries, and batch/fallback behavior.
3. Instantiate the workflow with a test `Config` and verify its agent list and
   graph structure before making model calls.
4. Run a short prompt manually, then generate a tiny benchmark dataset and
   check that each evaluator's `extract_gt_from_answer` output has the expected
   schema.
5. Run a benchmark and inspect the per-entry serialized states and analysis
   dashboard, including the Output Visualizer, when debugging trace alignment.

Keep external model calls out of fast unit tests where possible. Use mocks or
fakes for LLMs and tools, and use a small integration test when validating the
complete workflow against a real provider.

## Extension summary

| Extension point | Main responsibility | Key method |
|---|---|---|
| `Agent` | Perform one workflow step and append its result to state | `core(state, llm)` |
| `Evaluator` | Score candidates, compare answers with reference data, and extract benchmark ground truth | `_eval`, `_batch_eval`, `_gt_eval`, `extract_gt_from_answer` |
| `Workflow` | Instantiate agents and connect graph nodes, entry point, routes, and terminal edges | `initialize(config, graph)` |
