import json
from collections.abc import Callable
from pathlib import Path

from arco import workflows
from arco.core import Config, WorkflowFactory
from arco.data import BenchmarkDataset, BenchmarkEntry, Trace, TraceElement


def generate_benchmark(
    config_path: str,
    prompts_path: str,
    save_path: str,
    run_visualization_logic: Callable,
):
    """Generate a benchmark dataset by running a workflow on each prompt.

    Args:
        config_path: Path to a run-config YAML (the "giga model" config).
        prompts_path: Path to a JSON file containing a list of prompt entries.
            Each entry must have a ``"prompt"`` string and may include a
            ``"difficulty"`` integer (defaults to 1).
        save_path: Path where the generated benchmark JSON will be written.
        run_visualization_logic: Callable that receives ``workflow.stream(config=…)``
            and returns the final :class:`~arco.core.State`.
    """
    workflows.load_library_workflows()

    with open(prompts_path) as f:
        raw: list = json.load(f)

    # Normalise: accept both flat string lists and dict lists
    prompts: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            prompts.append({"id": len(prompts), "prompt": item, "difficulty": 1})
        else:
            prompts.append(
                {
                    "id": item.get("id", len(prompts)),
                    "prompt": item["prompt"],
                    "difficulty": item.get("difficulty", 1),
                }
            )

    default_config = Config.from_yaml(config_path)
    workflow = WorkflowFactory.get(config=default_config)

    prompts_len = len(prompts)
    yield {"event": "started", "total": prompts_len}

    entries: list[BenchmarkEntry] = []
    for index, entry_def in enumerate(prompts):
        entry_id = entry_def["id"]
        prompt = entry_def["prompt"]
        difficulty = entry_def["difficulty"]
        yield {
            "event": "prompt_start",
            "index": index,
            "prompt": prompt[:80],
            "id": entry_id,
            "total": prompts_len,
        }

        config = default_config.update_prompt(prompt)
        resulting_state = run_visualization_logic(workflow.stream(config=config))

        if resulting_state is None:
            yield {
                "event": "prompt_error",
                "index": entry_id,
                "message": "Workflow returned None",
            }
            continue

        trace_elements: list[TraceElement] = []
        evaluators = workflow.get_evaluators()

        for answer in resulting_state.answers:
            evaluator = evaluators.get(answer.agent_id)
            gt_data = evaluator.extract_gt_from_answer(answer) if evaluator else {}
            trace_elements.append(
                TraceElement(agent_type=answer.agent_id, data=gt_data)
            )

        entry = BenchmarkEntry(
            prompt=prompt,
            trace=Trace(trace_list=trace_elements),
            id=entry_id,
            difficulty=difficulty,
        )
        entries.append(entry)
        yield {
            "event": "prompt_done",
            "index": entry_id,
            "trace_len": len(trace_elements),
        }

    dataset = BenchmarkDataset(entries=entries)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save(save_path)

    yield {
        "event": "completed",
        "path": save_path,
        "entries": len(entries),
    }


__all__ = ["generate_benchmark"]
