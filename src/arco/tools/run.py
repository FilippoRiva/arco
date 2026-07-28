from pathlib import Path
from typing import TYPE_CHECKING

from arco import workflows
from arco.core import Config, WorkflowFactory
from arco.logs import initialize as init_logging

if TYPE_CHECKING:
    from arco.core import Workflow


def run_from_config(yaml_path: str, log_level: str | None = None):
    workflows.load_library_workflows()
    config = Config.from_yaml(yaml_path)
    workflow = WorkflowFactory.get(config=config)
    yield config
    yield workflow
    yield from run(config=config, workflow=workflow, log_level=log_level)


def run(config: Config, workflow: Workflow, log_level: str | None = None):

    init_logging(config.run_id, log_dir=Path(config.save_dir) / "logs", level=log_level)

    yield from workflow.stream()
