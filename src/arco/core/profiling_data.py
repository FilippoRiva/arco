from copy import deepcopy
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True, slots=True)
class ProfilingData:
    """Profiling metrics for a single agent execution step.

    :ivar total_time: Total wall-clock time in seconds.
    :ivar llm_time: Time spent in LLM API calls in seconds.
    :ivar energy_consumed_kwh: Total energy consumed in kWh.
    :ivar cpu_energy_kwh: CPU energy in kWh.
    :ivar gpu_energy_kwh: GPU energy in kWh.
    :ivar ram_energy_kwh: RAM energy in kWh.
    :ivar emissions_kg_co2: CO2 emissions in kg.
    """

    total_time: float | None = None
    llm_time: float | None = None
    energy_consumed_kwh: float | None = None
    cpu_energy_kwh: float | None = None
    gpu_energy_kwh: float | None = None
    ram_energy_kwh: float | None = None
    emissions_kg_co2: float | None = None

    def __add__(self, other: ProfilingData):
        return self.add(**other.as_dict())

    def add(
        self,
        *,
        total_time: float | None = None,
        llm_time: float | None = None,
        energy_consumed_kwh: float | None = None,
        cpu_energy_kwh: float | None = None,
        gpu_energy_kwh: float | None = None,
        ram_energy_kwh: float | None = None,
        emissions_kg_co2: float | None = None,
    ) -> ProfilingData:
        return replace(
            self,
            total_time=(self.total_time or 0) + (total_time or 0),
            llm_time=(self.llm_time or 0) + (llm_time or 0),
            energy_consumed_kwh=(self.energy_consumed_kwh or 0)
            + (energy_consumed_kwh or 0),
            cpu_energy_kwh=(self.cpu_energy_kwh or 0) + (cpu_energy_kwh or 0),
            gpu_energy_kwh=(self.gpu_energy_kwh or 0) + (gpu_energy_kwh or 0),
            ram_energy_kwh=(self.ram_energy_kwh or 0) + (ram_energy_kwh or 0),
            emissions_kg_co2=(self.emissions_kg_co2 or 0) + (emissions_kg_co2 or 0),
        )

    def as_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> ProfilingData:
        return deepcopy(self)


__all__ = ["ProfilingData"]
