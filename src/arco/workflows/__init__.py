def load_library_workflows():
    import importlib

    library_workflows_modules = ["sales"]
    for module in library_workflows_modules:
        importlib.import_module(f".{module}", package="arco.workflows")
