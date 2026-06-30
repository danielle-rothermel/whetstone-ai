"""Light-mode matplotlib helpers for analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_light_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "#333333",
            "grid.color": "#cccccc",
            "font.size": 11,
        }
    )


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def annotate_bars(ax, bars, *, fmt: str = "{:.0%}") -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
