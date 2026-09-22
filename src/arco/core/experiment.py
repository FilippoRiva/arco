"""Experiment catalog used to connect generation, benchmarking, and analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A named, reproducible connection between benchmark artifacts."""

    id: str
    description: str
    workflow: str
    generation_config: str
    prompts: str
    ground_truth: str
    benchmark_config: str
    output_dir: str
    tags: tuple[str, ...] = ()
    root_dir: Path = field(repr=False, compare=False, default_factory=Path.cwd)

    def path(self, value: str) -> Path:
        """Resolve a catalog path relative to the repository root."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root_dir / path

    @property
    def generation_config_path(self) -> Path:
        return self.path(self.generation_config)

    @property
    def prompts_path(self) -> Path:
        return self.path(self.prompts)

    @property
    def ground_truth_path(self) -> Path:
        return self.path(self.ground_truth)

    @property
    def benchmark_config_path(self) -> Path:
        return self.path(self.benchmark_config)

    @property
    def output_dir_path(self) -> Path:
        return self.path(self.output_dir)

    def metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata suitable for benchmark outputs."""
        return {
            "id": self.id,
            "description": self.description,
            "workflow": self.workflow,
            "generation_config": str(self.generation_config_path),
            "prompts": str(self.prompts_path),
            "ground_truth": str(self.ground_truth_path),
            "benchmark_config": str(self.benchmark_config_path),
            "output_dir": str(self.output_dir_path),
            "tags": list(self.tags),
        }


class ExperimentCatalog:
    """Load and retrieve named experiments from ``config/catalog.yaml``."""

    def __init__(self, experiments: dict[str, ExperimentSpec], root_dir: Path):
        self._experiments = experiments
        self.root_dir = root_dir

    @classmethod
    def load(cls, path: str | Path = "config/catalog.yaml") -> ExperimentCatalog:
        catalog_path = Path(path).expanduser().resolve()
        with catalog_path.open() as file:
            raw = yaml.safe_load(file) or {}

        # Catalog paths are repository-relative by convention. A catalog under
        # config/ therefore resolves paths from its parent directory.
        root_dir = catalog_path.parent.parent
        experiments: dict[str, ExperimentSpec] = {}
        for experiment_id, values in raw.get("experiments", {}).items():
            generation = values.get("generation", {})
            benchmark = values.get("benchmark", {})
            experiments[experiment_id] = ExperimentSpec(
                id=experiment_id,
                description=values.get("description", ""),
                workflow=values["workflow"],
                generation_config=generation["config"],
                prompts=generation["prompts"],
                ground_truth=generation["output"],
                benchmark_config=benchmark["config"],
                output_dir=benchmark["output_dir"],
                tags=tuple(values.get("tags", [])),
                root_dir=root_dir,
            )
        return cls(experiments, root_dir=root_dir)

    def get(self, experiment_id: str) -> ExperimentSpec:
        try:
            return self._experiments[experiment_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._experiments)) or "none"
            raise KeyError(
                f"Unknown experiment {experiment_id!r}. Available: {available}"
            ) from exc

    def all(self) -> dict[str, ExperimentSpec]:
        return dict(self._experiments)


__all__ = ["ExperimentCatalog", "ExperimentSpec"]
