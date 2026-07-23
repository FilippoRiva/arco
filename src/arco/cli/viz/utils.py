from rich.status import Status


def execute_chart_code(df, chart_config, code):
    with Status("Waiting for image to render", spinner="dots", refresh_per_second=8) as status:
        import matplotlib
        for backend in ("TkAgg", "Qt5Agg", "GTK3Agg"):
            try:
                matplotlib.use(backend)
                import matplotlib.pyplot as plt
                break
            except ImportError:
                plt = None

        if plt is None:
            print(
                "No interactive backend available. Install one of: python3-tkinter (system), PyQt5 (pip), or PyGObject (pip)")
            return

        namespace = {
            "data_df": df,
            "config": chart_config,
            "plt": plt,
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
        }
        status.update("Waiting for visualization window to close")
        exec(code, namespace)  # noqa: S102
