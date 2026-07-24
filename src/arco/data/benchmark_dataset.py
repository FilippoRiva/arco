import json
from collections.abc import Iterator
from dataclasses import dataclass

from arco.core import AgentType
from arco.core.profiling_data import ProfilingData


@dataclass(frozen=True)
class BenchmarkSummary:
    completion_percentage: float
    ppls: list[float]
    scores: list[float]
    agents: list[AgentType]
    profiling_datas: list[ProfilingData]


@dataclass(frozen=True)
class BenchmarkDataset:
    entries: list[BenchmarkEntry]

    @classmethod
    def from_json(cls, json_path) -> BenchmarkDataset:
        with open(json_path) as f:
            json_data = json.load(f)

        entries = []
        for entry in json_data:
            entries.append(BenchmarkEntry.from_dict(entry_dict=entry))

        return cls(entries=entries)

    def __iter__(self) -> Iterator[BenchmarkEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class BenchmarkEntry:
    prompt: str
    trace: Trace
    id: int
    difficulty: int

    @classmethod
    def from_dict(
        cls, entry_dict: dict[str, str | int | float | dict]
    ) -> BenchmarkEntry:
        return cls(
            prompt=entry_dict["prompt"],
            id=int(entry_dict["id"]),
            difficulty=int(entry_dict["difficulty"]),
            trace=Trace.from_trace_data(entry_dict["trace"]),
        )


@dataclass(frozen=True)
class Trace:
    trace_list: list[TraceElement]

    @classmethod
    def from_trace_data(
        cls, trace_list_data: list[dict[str, str | int | float | dict]]
    ):
        trace_list = []
        for trace_element in trace_list_data:
            agent_type = AgentType(trace_element["agent_type"])
            data = trace_element["data"]
            trace_list.append(TraceElement(agent_type=agent_type, data=data))
        return cls(trace_list=trace_list)

    def __getitem__(self, idx: int) -> TraceElement:
        return self.trace_list[idx]

    def __len__(self):
        return len(self.trace_list)

    def __iter__(self) -> Iterator[TraceElement]:
        return iter(self.trace_list)


@dataclass(frozen=True)
class TraceElement:
    agent_type: AgentType
    data: dict[str, str | int | float | dict]
