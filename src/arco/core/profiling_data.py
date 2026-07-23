from dataclasses import dataclass, replace, asdict


@dataclass(frozen=True)
class ProfilingData:
    total_time: float | None = None
    llm_time: float | None = None
    energy_consumed_kwh: float | None = None
    cpu_energy_kwh: float | None = None
    gpu_energy_kwh: float | None = None
    ram_energy_kwh: float | None = None
    emissions_kg_co2: float | None = None

    def __add__(self, other: ProfilingData):
        return self.add(**other.as_dict())

    def add(self,
            *,
            total_time: float | None = None,
            llm_time: float | None = None,
            energy_consumed_kwh: float | None = None,
            cpu_energy_kwh: float | None = None,
            gpu_energy_kwh: float | None = None,
            ram_energy_kwh: float | None = None,
            emissions_kg_co2: float | None = None,
            ):
        return replace(
            self,
            total_time=(self.total_time or 0) + (total_time or 0),
            llm_time=(self.llm_time or 0) + (llm_time or 0),
            energy_consumed_kwh=(self.energy_consumed_kwh or 0) + (energy_consumed_kwh or 0),
            cpu_energy_kwh=(self.cpu_energy_kwh or 0) + (cpu_energy_kwh or 0),
            gpu_energy_kwh=(self.gpu_energy_kwh or 0) + (gpu_energy_kwh or 0),
            ram_energy_kwh=(self.ram_energy_kwh or 0) + (ram_energy_kwh or 0),
            emissions_kg_co2=(self.emissions_kg_co2 or 0) + (emissions_kg_co2 or 0),
        )

    def as_dict(self):
        return asdict(self)
