def load_workflows() -> list[str]:
    # Imports
    import importlib

    from arco.core import WorkflowFactory

    # Silent load of library defined workflows (it loads everything if dependencies are available)
    library_workflows_modules: list[str] = ["sales"]
    for module in library_workflows_modules:
        try:
            importlib.import_module(f".{module}", package="arco.workflows")
        except ModuleNotFoundError:
            pass

    # Return all the available workflows
    return list(WorkflowFactory.all().keys())
