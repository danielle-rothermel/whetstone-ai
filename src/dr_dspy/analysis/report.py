"""Rich terminal reporting for analysis scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from dr_dspy.analysis.figures import FigureRun


class AnalysisReporter:
    def __init__(self, *, title: str, script_name: str) -> None:
        self.console = Console(record=True, width=120)
        self.title = title
        self.script_name = script_name

    def print_header(self) -> None:
        self.console.print()
        self.console.print(
            Panel(
                Text(self.title, style="bold white"),
                subtitle=self.script_name,
                border_style="blue",
                padding=(1, 2),
            )
        )

    def print_run_context(
        self,
        *,
        experiment_names: Sequence[str],
        row_count: int,
        limit: int | None,
        require_score: bool | None = None,
    ) -> None:
        experiments = ", ".join(experiment_names)
        limit_text = str(limit) if limit is not None else "none"
        body = (
            f"[bold]Experiments[/]  {experiments}\n"
            f"[bold]Rows loaded[/]   {row_count}\n"
            f"[bold]Row limit[/]     {limit_text}"
        )
        if require_score is not None:
            score_filter = (
                "success only" if require_score 
                else "include unscored"
            )
            body += f"\n[bold]Score filter[/]  {score_filter}"
        self.console.print(
            Panel(
                body,
                title="Run context",
                border_style="cyan",
                padding=(0, 2),
            )
        )

    def section(self, label: str) -> None:
        self.console.print()
        self.console.print(Rule(label, style="bold magenta"))
        self.console.print()

    def print_table(self, table: Table) -> None:
        self.console.print(table)

    def print_note(self, message: str) -> None:
        self.console.print(
            Panel(message, title="Note", border_style="yellow", padding=(0, 2))
        )

    def finish(
        self,
        *,
        output_run: FigureRun,
        csv_paths: Sequence[Path],
        md_paths: Sequence[Path],
        figure_paths: Sequence[Path],
    ) -> Path:
        artifacts_dir = output_run.artifacts_directory.resolve()
        figures_dir = output_run.figures_directory.resolve()
        tabular_lines = [
            f"  {path.resolve()}"
            for path in (*csv_paths, *md_paths)
            if path.exists()
        ]
        figure_lines = [
            f"  {path.resolve()}" for path in figure_paths if path.exists()
        ]
        body_parts = ["[bold]Tabular artifacts[/bold]"]
        body_parts.extend(tabular_lines or ["  (none)"])
        body_parts.extend(["", "[bold]Figure files[/bold]"])
        body_parts.extend(figure_lines or ["  (none)"])
        self.console.print()
        self.console.print(
            Panel(
                "\n".join(body_parts),
                title="Written outputs",
                border_style="green",
                padding=(0, 2),
            )
        )
        self.console.print()
        self.console.print(
            Panel(
                str(artifacts_dir),
                title="Artifacts folder — open or copy this path",
                border_style="bold green",
                padding=(0, 2),
            )
        )
        self.console.print(
            Panel(
                str(figures_dir),
                title="Figures folder — open or copy this path",
                border_style="bold yellow",
                padding=(0, 2),
            )
        )
        log_path = output_run.run_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console.print(
            Panel(
                str(log_path.resolve()),
                title="Run log (HTML) — open in browser",
                border_style="bold blue",
                padding=(0, 2),
            )
        )
        self.console.save_html(str(log_path))
        self.console.print()
        return log_path
