import time

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


class RunStatusPanel:
    def __init__(self, compact: bool = False):
        self.compact = compact
        self.status: str = ""
        self.init_time: float = time.time()
        self.node_start_time: float | None = None
        self.spinner: Spinner = Spinner("dots")
        self.stopped: bool = False
        self.stream_buffer: str = ""
        self.tool_events: list[Text] = []
        self.active_node: str = ""

    def start(self):
        self.stopped = False

    def stop(self):
        self.stopped = True

    def set(self, status: str, start_time: float | None = None):
        self.status = status
        self.node_start_time = start_time

    def set_state(self, state: str):
        """Update the live status text while preserving the node timer."""
        self.status = state

    def append_stream(self, text: str):
        self.stream_buffer += text

    def append_tool_event(self, event: Text):
        """Add a persistent tool activity line for the active node."""
        self.tool_events.append(event)

    def clear_activity(self):
        self.stream_buffer = ""
        self.tool_events.clear()
        self.active_node = ""

    def clear_stream(self):
        self.stream_buffer = ""
        self.active_node = ""

    def __rich__(self) -> str | RenderableType:
        if self.stopped:
            return ""

        total_elapsed = time.time() - self.init_time
        if self.compact:
            # Keep the running view to one quiet line. Completed work is
            # printed by display_workflow; this is only the live activity hint.
            line = Text(self.status or "Starting", style="cyan")
            if self.node_start_time is not None:
                line.append(
                    f"  {time.time() - self.node_start_time:.1f}s",
                    style="dim",
                )
            self.spinner.update(text="")

            grid = Table.grid(padding=(0, 1), expand=False)
            grid.add_column()
            grid.add_column()
            grid.add_row(self.spinner, line)
            live_content: list[RenderableType] = [grid]
            if self.tool_events:
                live_content.append(Text(""))
                live_content.extend(self.tool_events)
            if self.stream_buffer:
                live_content.extend(
                    [
                        Text(""),
                        Markdown(self.stream_buffer),
                    ]
                )
            return Group(*live_content)

        lines: list[RenderableType] = []

        text = f"[yellow]{self.status}[/yellow]"

        if self.node_start_time is not None:
            elapsed = time.time() - self.node_start_time
            text += f" [dim]Node time : {elapsed:.1f}s[/dim] "

        text += f" [dim]Total time : {total_elapsed:.1f}s[/dim]"

        self.spinner.update(text=text)
        lines.append(self.spinner)

        if self.stream_buffer:
            lines.append("")
            lines.append(Markdown(self.stream_buffer))

        return Panel(
            Group(*lines),
            title=f"[cyan]{self.active_node}[/cyan]" if self.active_node else "",
            border_style="cyan" if self.active_node else "dim",
        )
