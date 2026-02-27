from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from helper.plot.color import Color, QualColors
from helper.validator import validator

##### config start #####
FIG_SIZE = (8, 6)
LINE_WIDTH = 2.0
SCATTER_MULTIPLE = 3.0

FONT_SIZE_AXIS_LABEL = 14
FONT_SIZE_TITLE = 16
FONT_SIZE_LEGEND = 10

Y_TITLE = 1.02
##### config end #####

LINE_STYLES = ["-", "--", ":", "-."]

LEGEND_LOC = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
    "outside",
]

LEGEND_LOC_INSIDE = Literal[
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


class Legend:
    handles: list[Line2D | Patch]
    labels: list[str]
    loc: LEGEND_LOC_INSIDE
    fontsize: float
    n_col: int
    bbox_to_anchor: tuple[float, float] | None = None
    spine_width: float | None = None

    def __init__(
        self,
        handle_type: Literal["round", "square"],
        handle_colors: tuple[Color, ...] | QualColors,
        labels: list[str],
        loc: LEGEND_LOC_INSIDE,
        fontsize: float,
        n_col: int,
        bbox_to_anchor: tuple[float, float] | None = None,
        spine_width: float | None = None,
    ):
        if isinstance(handle_colors, QualColors):
            colors = handle_colors.get_n(len(labels))
        else:
            colors = handle_colors[:]

        if len(colors) != len(labels):
            raise ValueError(
                f"The length of `handle_colors` ({len(colors)}) must be equal to the length of `labels` ({len(labels)})."
            )

        handles: list[Line2D | Patch] = []
        for color in colors:
            if handle_type == "round":
                handles.append(Line2D([], [], marker="o", linestyle="", markersize=fontsize * 2 / 3, color=color))  # type: ignore[arg-type]
            else:
                handles.append(Patch(facecolor=color, edgecolor=color))  # type: ignore[arg-type]

        self.handles = handles
        self.labels = labels
        self.loc = loc
        self.fontsize = fontsize
        self.n_col = n_col
        self.bbox_to_anchor = bbox_to_anchor
        self.spine_width = spine_width

    def apply(self, target: Figure | Axes) -> None:
        legend = target.legend(
            handles=self.handles,
            labels=self.labels,
            loc=self.loc,
            bbox_to_anchor=self.bbox_to_anchor,
            ncol=self.n_col,
            fontsize=self.fontsize,
        )
        if self.spine_width is not None:
            legend.get_frame().set_linewidth(self.spine_width)


def get_title(plot_title: str, title_text: str) -> str:
    if plot_title and title_text:
        return f"{plot_title} - {title_text}"
    else:
        return plot_title + title_text


def calculate_x_group(y: list[list[Any]], spacing: float, group_spacing: float) -> list[list[float]]:
    x: list[list[float]] = []
    x_group_start = 0.0

    for y_group in y:
        x_group = [x_group_start + j * spacing for j in range(len(y_group))]
        x.append(x_group)
        x_group_start = x_group[-1] + group_spacing

    return x


def diameter_to_area(diameter: float) -> float:
    return np.pi * 0.25 * diameter**2


def calculate_scatter_size(line_width: float, multiple: float = SCATTER_MULTIPLE) -> float:
    return line_width**2 * multiple**2


@validator.constraints("n_epochs", "x > 1")
@validator.constraints("n_warmup_epochs", "x >= 0")
def set_axis_epoch(ax: plt.Axes, n_epochs: int, n_warmup_epochs: int = 0, shift_left_xlim: float = 0) -> range:
    if n_warmup_epochs > 0:
        tick_position = np.linspace(n_warmup_epochs, n_warmup_epochs + n_epochs, 6, dtype=int)
    else:
        tick_position = np.linspace(1, n_epochs, 6, dtype=int)

    ax.set_xlim(1 + shift_left_xlim, n_epochs + n_warmup_epochs)
    ax.set_xticks(tick_position)
    ax.set_xticklabels([str(t_p - n_warmup_epochs) for t_p in tick_position])
    ax.get_xticklabels()[-1].set_ha("right")  # type: ignore[attr-defined]

    return range(1, n_epochs + n_warmup_epochs + 1)


def save_plot(fig: plt.Figure, dir_: Path, file_name: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(dir_ / file_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
