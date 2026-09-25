import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pandas as pd

from arco import workflows
from arco.core import (
    Config,
    State,
    Workflow,
    WorkflowFactory,
    evaluate_state_with_benchmark_entry,
)
from arco.data import BenchmarkDataset
from arco.logs import initialize as init_logging

_RUN_CSV_COLUMNS = [
    "entry_id",
    "run_id",
    "run_fingerprint",
    "changes",
    "state",
]


def _run_output_name(name: str) -> str:
    """Convert a run label to a safe flat CSV filename stem."""
    if not name or any(char in name for char in ("/", "\\", "\n", "\r", "\0")):
        raise ValueError(f"Unsafe benchmark run name: {name!r}")
    output_name = name.replace(" ", "_")
    if output_name in {".", ".."}:
        raise ValueError(f"Unsafe benchmark run name: {name!r}")
    return output_name


def _prepare_input_snapshot(source: str, destination: Path) -> tuple[Path, bool]:
    """Create an immutable input snapshot, or validate an existing one."""
    source_path = Path(source).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if source_path.is_file() and hashlib.sha256(source_path.read_bytes()).digest() != (
            hashlib.sha256(destination.read_bytes()).digest()
        ):
            raise ValueError(
                f"Benchmark input {source_path} differs from the existing snapshot "
                f"{destination}. Use a new benchmark name or remove the old output "
                "directory to start a new experiment."
            )
        return destination, False

    if not source_path.is_file():
        raise FileNotFoundError(f"Benchmark input not found: {source_path}")

    temporary_path = destination.with_name(f".{destination.name}.copying")
    shutil.copyfile(source_path, temporary_path)
    os.replace(temporary_path, destination)
    return destination, True


def _run_fingerprint(
    config: Config, changes: Any, benchmark_dataset: BenchmarkDataset
) -> str:
    """Hash the effective run configuration and ordered ground-truth dataset."""
    config_payload: dict[str, Any] = {}
    for config_field in fields(Config):
        name = config_field.name
        if name in {"prompt", "run_id", "save_dir", "config_path"}:
            continue
        value = getattr(config, name)
        if name == "agent_configs":
            value = {
                str(agent_type): asdict(agent_config)
                for agent_type, agent_config in value.items()
            }
        config_payload[name] = value

    payload = {
        "config": config_payload,
        "changes": changes,
        "dataset": [entry.to_dict() for entry in benchmark_dataset],
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _repair_truncated_checkpoint(csv_path: Path) -> bool:
    """Drop a partial trailing CSV record left by an interrupted append."""
    with csv_path.open("r+b") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        if size == 0:
            return True
        file.seek(-1, os.SEEK_END)
        if file.read(1) == b"\n":
            return False
        file.seek(0)
        contents = file.read()
        last_newline = contents.rfind(b"\n")
        if last_newline < 0:
            # A process may have stopped while the header was being written.
            file.truncate(0)
            file.flush()
            os.fsync(file.fileno())
            return True
        file.truncate(last_newline + 1)
        file.flush()
        os.fsync(file.fileno())
        return True


def _load_checkpoint(
    csv_path: Path,
    *,
    expected_entry_ids: set[int],
    fingerprint: str,
) -> tuple[dict[int, dict[str, Any]], bool]:
    """Load completed entry rows and recover an interrupted trailing append."""
    if not csv_path.exists():
        return {}, False

    recovered = _repair_truncated_checkpoint(csv_path)
    if csv_path.stat().st_size == 0:
        return {}, recovered
    try:
        dataframe = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read benchmark checkpoint {csv_path}") from exc

    checkpoint_columns = set(dataframe.columns)
    if checkpoint_columns != set(_RUN_CSV_COLUMNS):
        missing_columns = set(_RUN_CSV_COLUMNS).difference(checkpoint_columns)
        unexpected_columns = checkpoint_columns.difference(_RUN_CSV_COLUMNS)
        raise ValueError(
            f"Checkpoint {csv_path} has an unsupported schema; "
            f"missing columns: {sorted(missing_columns)}, "
            f"unexpected columns: {sorted(unexpected_columns)}"
        )

    fingerprints = set(dataframe["run_fingerprint"].dropna().astype(str))
    if fingerprints and fingerprints != {fingerprint}:
        raise ValueError(
            f"Checkpoint {csv_path} was created from different benchmark inputs. "
            "Use a new benchmark name or remove the stale CSV before rerunning."
        )

    completed: dict[int, dict[str, Any]] = {}
    for row in dataframe.to_dict(orient="records"):
        try:
            entry_id = int(row["entry_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid entry_id in checkpoint {csv_path}") from exc
        if entry_id not in expected_entry_ids:
            raise ValueError(
                f"Unexpected entry_id {entry_id} in checkpoint {csv_path}"
            )
        if entry_id in completed:
            raise ValueError(f"Duplicate entry_id {entry_id} in checkpoint {csv_path}")
        if str(row["run_fingerprint"]) != fingerprint:
            raise ValueError(
                f"Checkpoint row for entry {entry_id} has a mismatched fingerprint"
            )
        try:
            json.loads(row["changes"])
            json.loads(row["state"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Malformed changes/state JSON for entry {entry_id} in {csv_path}"
            ) from exc
        completed[entry_id] = row

    return completed, recovered


def _append_checkpoint_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Durably append one completed entry to its run CSV checkpoint."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_RUN_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


def benchmark_from_config(
    config_path: str,
    dataset_path: str,
    id: str | None,
    save_dir: str,
    logging_level: str | None,
    experiment_metadata: dict[str, Any] | None = None,
):
    available_workflows = workflows.load_workflows()
    if len(available_workflows) == 0:
        yield {"event": "error", "message": "No workflow available"}
        return

    start_time = time.time()

    # Prepare the experiment output folder, then snapshot inputs before loading
    # them so the benchmark always executes from the archived copies.
    benchmark_id = id or Path(config_path).stem
    benchmark_save_folder = Path(save_dir) / benchmark_id
    runs_folder = benchmark_save_folder / "runs"
    if runs_folder.exists() and any(path.is_dir() for path in runs_folder.iterdir()):
        raise ValueError(
            f"Legacy per-run subdirectories found under {runs_folder}. "
            "This benchmark format uses flat CSV files; move or remove the old "
            "runs directory before starting."
        )
    benchmark_save_folder.mkdir(parents=True, exist_ok=True)

    config_snapshot, config_copied = _prepare_input_snapshot(
        config_path, benchmark_save_folder / "benchmark_config.yaml"
    )
    dataset_snapshot, dataset_copied = _prepare_input_snapshot(
        dataset_path, benchmark_save_folder / "dataset.json"
    )
    yield {
        "event": "benchmark_inputs_snapshotted",
        "config_path": config_snapshot,
        "dataset_path": dataset_snapshot,
        "config_copied": config_copied,
        "dataset_copied": dataset_copied,
    }

    default_config = Config.from_yaml(str(config_snapshot))
    # Storage is opt-in when unspecified; disable CodeCarbon for benchmark runs
    # so they do not create energy-tracking output directories.
    if default_config.enable_storage is None:
        default_config = default_config.set(enable_storage=False)
    default_config = default_config.set(enable_codecarbon=False)
    workflow = WorkflowFactory.get(config=default_config)
    benchmark_dataset = BenchmarkDataset.from_json(str(dataset_snapshot))
    if len(benchmark_dataset) == 0:
        raise ValueError("Benchmark dataset must contain at least one entry")

    init_logging(benchmark_id, log_dir=benchmark_save_folder, level=logging_level)

    # Load run configurations (one per specified run in the archived config)
    list_of_run_configs = default_config.generate_benchmark_configs(
        str(config_snapshot)
    )
    yield {"event": "run_configs_loaded", "configs": list_of_run_configs}

    # Run each configuration. Each completed entry is durably appended to its
    # run CSV, which serves as both the final artifact and the resume checkpoint.
    entry_ids = [entry.id for entry in benchmark_dataset]
    expected_entry_ids = set(entry_ids)
    if len(expected_entry_ids) != len(entry_ids):
        raise ValueError("Benchmark entry IDs must be unique to support resuming")

    run_names = [_run_output_name(run["name"]) for run in list_of_run_configs]
    if len(set(run_names)) != len(run_names):
        raise ValueError(
            "Benchmark run names collide after converting them to flat CSV filenames"
        )

    for run_config_dict, run_name in zip(list_of_run_configs, run_names, strict=True):
        run_csv_path = runs_folder / f"{run_name}.csv"
        run_config: Config = run_config_dict["config"]
        changes = run_config_dict["changes"]
        fingerprint = _run_fingerprint(run_config, changes, benchmark_dataset)
        completed_rows, recovered_partial_row = _load_checkpoint(
            run_csv_path,
            expected_entry_ids=expected_entry_ids,
            fingerprint=fingerprint,
        )

        yield {
            "event": "benchmark_start",
            "name": run_config_dict["name"],
            "description": run_config_dict["description"],
            "changes": changes,
        }
        if recovered_partial_row:
            yield {"event": "benchmark_checkpoint_recovered", "path": run_csv_path}

        cached_count = len(completed_rows)
        if cached_count == len(benchmark_dataset):
            yield {
                "event": "benchmark_already_exists",
                "path": run_csv_path,
                "cached_entries": cached_count,
                "total_entries": len(benchmark_dataset),
            }
        else:
            if cached_count:
                yield {
                    "event": "benchmark_resume",
                    "path": run_csv_path,
                    "cached_entries": cached_count,
                    "total_entries": len(benchmark_dataset),
                    "remaining_entries": len(benchmark_dataset) - cached_count,
                }
            else:
                yield {
                    "event": "benchmark_fresh_start",
                    "total_entries": len(benchmark_dataset),
                }

            for event in benchmark(
                workflow=workflow,
                name=run_config_dict["name"],
                config=run_config,
                changes=changes,
                benchmark_dataset=benchmark_dataset,
                completed_entry_ids=set(completed_rows),
                run_fingerprint=fingerprint,
            ):
                if event["event"] == "entry_result":
                    row = event["row"]
                    _append_checkpoint_row(run_csv_path, row)
                    completed_rows[int(row["entry_id"])] = row
                    yield {
                        "event": "benchmark_entry_checkpoint",
                        "entry_id": int(row["entry_id"]),
                        "completed_entries": len(completed_rows),
                        "total_entries": len(benchmark_dataset),
                        "path": run_csv_path,
                    }
                else:
                    yield event

            if set(completed_rows) != expected_entry_ids:
                raise RuntimeError(
                    f"Run {run_name!r} ended without checkpointing every benchmark entry"
                )
            yield {"event": "benchmark_run_save", "path": run_csv_path}

    bench_metadata = {
        "experiment": experiment_metadata,
        "benchmark_run": config_path,
        "benchmark_config_snapshot": str(config_snapshot),
        "dataset_path": str(dataset_snapshot),
        "dataset_source": dataset_path,
        "total_runtime": time.time() - start_time,
    }
    with open(benchmark_save_folder / "bench_metadata.json", "w") as f:
        json.dump(bench_metadata, f)
    yield {
        "event": "benchmark_complete",
        "output_dir": benchmark_save_folder,
        "metadata_path": benchmark_save_folder / "bench_metadata.json",
    }



def benchmark(
    workflow: Workflow,
    name: str,
    config: Config,
    changes: Any,
    benchmark_dataset: BenchmarkDataset,
    *,
    completed_entry_ids: set[int] | None = None,
    run_fingerprint: str,
):
    """Execute missing entries and emit a durable-row candidate per success."""
    completed_entry_ids = completed_entry_ids or set()

    for iteration, entry in enumerate(benchmark_dataset, start=1):
        if entry.id in completed_entry_ids:
            continue

        yield {
            "event": "test_case_start",
            "iteration": iteration,
            "max_iteration": len(benchmark_dataset),
            "entry_id": entry.id,
        }

        config_new_prompt: Config = config.update_prompt(entry.prompt)
        updated_config: Config = config_new_prompt.set(run_id=name + str(entry.id))
        resulting_state: State | None = None
        workflow_errors: list[str] = []
        for workflow_event in workflow.stream(config=updated_config):
            if workflow_event["event"] == "error":
                workflow_errors.append(str(workflow_event.get("message", "Unknown error")))
            elif workflow_event["event"] == "completed":
                resulting_state = workflow_event.get("state")
            yield {"event": "workflow_event", "workflow_event": workflow_event}

        if workflow_errors or resulting_state is None:
            details = "; ".join(workflow_errors) or "Workflow completed without a state"
            raise RuntimeError(f"Benchmark case {entry.id} failed: {details}")

        yield {"event": "test_case_evaluation_start"}
        benchmark_summary, updated_state = evaluate_state_with_benchmark_entry(
            resulting_state,
            entry,
            workflow.get_evaluators(),
            updated_config.default_provider_judge,
            updated_config.default_model_judge,
        )
        yield {"event": "test_case_evaluation_stop"}

        yield {"event": "test_case_stop", "evaluation_summary": benchmark_summary}

        # The full state is the sole per-entry result payload. Analysis derives
        # its compact trace and metrics from this serialized state.
        row = {
            "entry_id": entry.id,
            "run_id": updated_state.run_id,
            "run_fingerprint": run_fingerprint,
            "changes": json.dumps(
                changes, sort_keys=True, ensure_ascii=False, default=str
            ),
            "state": json.dumps(
                updated_state.to_dict(), default=str, ensure_ascii=False
            ),
        }

        # The caller appends this row to the checkpoint before continuing.
        yield {"event": "entry_result", "row": row}
