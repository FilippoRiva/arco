import time

from rich.spinner import Spinner


class RunStatusPanel:
    def __init__(self):
        self.status = ""
        self.init_time = time.time()
        self.node_start_time = None
        self.spinner = Spinner("dots")
        self.stopped = False

    def start(self):
        self.stopped = False

    def stop(self):
        self.stopped = True

    def set(self, status, start_time=None):
        self.status = status
        self.node_start_time = start_time

    def __rich__(self):
        if self.stopped:
            return ""

        text = f"[yellow]{self.status}[/yellow]"

        if self.node_start_time is not None:
            elapsed = time.time() - self.node_start_time
            text += f" [dim]Node time : {elapsed:.1f}s[/dim] "

        text += f"[dim]Total time : {time.time() - self.init_time:.1f}s[/dim]"

        self.spinner.update(text=text)
        return self.spinner
