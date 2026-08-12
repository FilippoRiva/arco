from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arco import workflows
from arco.core import Config, WorkflowFactory
from arco.logs import initialize as init_logging

if TYPE_CHECKING:
    from arco.core import Workflow


def initialize_workflow(
    yaml_path: str | None = None, workflow_name: str | None = None
) -> tuple[Config, Workflow]:
    workflows.load_workflows()
    if yaml_path:
        config = Config.from_yaml(yaml_path)
    elif workflow_name:
        config = Config(workflow=workflow_name)
    else:
        config = Config(workflow=get_workflow_list()[0])
    workflow = WorkflowFactory.get(config=config)
    return config, workflow


def get_workflow_list() -> list[str]:
    return workflows.load_workflows()


def run(
    config: Config, workflow: Workflow, log_level: str | None = None
) -> Generator[dict[str, Any]]:

    init_logging(config.run_id, log_dir=Path(config.save_dir) / "logs", level=log_level)

    yield from workflow.stream()
