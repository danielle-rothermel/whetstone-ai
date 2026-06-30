"""Output paths for analysis script artifacts and figures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FigureRun:
    script_name: str
    timestamp: str

    @classmethod
    def start(cls, script_name: str) -> FigureRun:
        return cls(
            script_name=script_name,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    @property
    def artifacts_directory(self) -> Path:
        return repo_root() / "artifacts" / self.script_name

    @property
    def directory(self) -> Path:
        return self.figures_directory

    @property
    def figures_directory(self) -> Path:
        return repo_root() / "figs" / self.script_name

    def artifact_path(
        self,
        artifact_name: str,
        *,
        suffix: str = "csv",
    ) -> Path:
        return self._build_path(
            self.artifacts_directory,
            artifact_name,
            suffix,
        )

    def path(self, figure_name: str, *, suffix: str = "png") -> Path:
        return self._build_path(self.figures_directory, figure_name, suffix)

    def run_log_path(self) -> Path:
        return self._build_path(self.figures_directory, "run", "html")

    def ensure_directories(self) -> None:
        self.artifacts_directory.mkdir(parents=True, exist_ok=True)
        self.figures_directory.mkdir(parents=True, exist_ok=True)

    def _build_path(
        self,
        root: Path,
        stem: str,
        suffix: str,
    ) -> Path:
        extension = suffix.lstrip(".")
        filename = f"{self.timestamp}_{stem}.{extension}"
        return root / filename
