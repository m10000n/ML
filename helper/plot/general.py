import itertools
from itertools import chain
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from helper.plot.color import Color, QualColors, SeqColors
from helper.plot.main import (
    FIG_SIZE,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_TITLE,
    LINE_STYLES,
    LINE_WIDTH,
    Y_TITLE,
    Legend,
    calculate_scatter_size,
    calculate_x_group,
    diameter_to_area,
    save_plot,
)


def plot_x_y(
    dir_: Path,
    file_name: str,
    x: list[int | float],
    y: list[list[int | float] | int | float],
    lines: bool = True,
    scatter: bool = True,
    horizontal: list[int | float] | int | float | None = None,
    vertical: list[int | float] | int | float | None = None,
    colors: list[Color] | Color | None = None,
    linestyle: bool = True,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    x_ticks: list[int | float] | None = None,
    x_lim: tuple[int | float, int | float] | None = None,
    y_lim: tuple[int | float, int | float] | None = None,
    line_labels: list[str] | str | None = None,
    scatter_labels: list[str] | str | None = None,
    horizontal_labels: list[str] | str | None = None,
    vertical_labels: list[str] | str | None = None,
    legend_loc: str = "best",
    legend_n_col: int = 1,
) -> None:
    if isinstance(y[0], (int, float)):
        y_: list[list[int | float]] = cast(list[list[int | float]], [y])
    else:
        y_ = cast(list[list[int | float]], y)

    n_y = len(y_)

    if horizontal is None:
        horizontal_ = []
    elif isinstance(horizontal, (int, float)):
        horizontal_ = [horizontal]
    else:
        horizontal_ = horizontal

    if vertical is None:
        vertical_ = []
    elif isinstance(vertical, (int, float)):
        vertical_ = [vertical]
    else:
        vertical_ = vertical

    if colors is None:
        colors_ = QualColors().get_n(n=n_y)
    elif isinstance(colors, Color):
        colors_ = (colors,)
    else:
        colors_ = tuple(colors)

    if line_labels is None:
        line_labels_: list[str | None] = [None] * n_y
    elif isinstance(line_labels, str):
        line_labels_ = cast(list[str | None], [line_labels])
    else:
        line_labels_ = cast(list[str | None], line_labels)

    if scatter_labels is None:
        scatter_labels_: list[str | None] = [None] * n_y
    elif isinstance(scatter_labels, str):
        scatter_labels_ = cast(list[str | None], [scatter_labels])
    else:
        scatter_labels_ = cast(list[str | None], scatter_labels)

    if horizontal_labels is None:
        horizontal_labels_: list[str | None] = [None] * len(horizontal_)
    elif isinstance(horizontal_labels, str):
        horizontal_labels_ = cast(list[str | None], [horizontal_labels])
    else:
        horizontal_labels_ = cast(list[str | None], horizontal_labels)

    if vertical_labels is None:
        vertical_labels_: list[str | None] = [None] * len(vertical_)
    elif isinstance(vertical_labels, str):
        vertical_labels_ = cast(list[str | None], [vertical_labels])
    else:
        vertical_labels_ = cast(list[str | None], vertical_labels)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.set_title(
        title,
        fontsize=FONT_SIZE_TITLE,
        y=Y_TITLE,
    )

    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    if x_lim is not None:
        ax.set_xlim(x_lim)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlabel(xlabel=x_label, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(ylabel=y_label, fontsize=FONT_SIZE_AXIS_LABEL)

    if lines:
        for i, (y__, color, line_label) in enumerate(zip(y_, colors_, line_labels_)):
            linestyle_ = LINE_STYLES[i % len(LINE_STYLES)] if linestyle else "-"
            ax.plot(x, y__, color=color, linestyle=linestyle_, zorder=i, label=line_label)

    if scatter:
        for i, (y__, color, scatter_label) in enumerate(zip(y_, colors_, scatter_labels_)):
            ax.scatter(
                x,
                y__,
                color=color,
                s=calculate_scatter_size(line_width=LINE_WIDTH),
                zorder=i + n_y,
                label=scatter_label,
            )

    for i, (h, label) in enumerate(zip(horizontal_, horizontal_labels_)):
        ax.axhline(h, color="gray", linestyle="--", linewidth=LINE_WIDTH / 2, zorder=i + 2 * n_y, label=label)

    for i, (v, label) in enumerate(zip(vertical_, vertical_labels_)):
        ax.axvline(
            v,
            color="gray",
            linestyle="--",
            linewidth=LINE_WIDTH / 2,
            zorder=i + 2 * n_y + len(horizontal_),
            label=label,
        )

    if any(label is not None for label in chain(line_labels_, scatter_labels_, horizontal_labels_, vertical_labels_)):
        if legend_loc == "outside":
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=legend_n_col)
        else:
            ax.legend(loc=legend_loc)

    save_plot(fig=fig, dir_=dir_, file_name=file_name)


def plot_ci_groups(
    dir_: Path,
    file_name: str,
    names: list[str],
    point: list[list[float]],
    ci_lower: list[list[float]],
    ci_upper: list[list[float]],
    spacing: float = 0.5,
    group_spacing: float = 1.0,
    marker_diameter: float = 7.0,
    capsize: float = 14.0,
    capthick: float = 2.0,
    elinewidth: float = 2.0,
    colors: tuple[Color, ...] | None = None,
    fig_size: tuple[float, float] = FIG_SIZE,
    title: str = "",
    names_fontsize: float = FONT_SIZE_AXIS_LABEL,
    names_rotation: int = 0,
    y_label: str = "",
    y_label_fontsize: float = FONT_SIZE_AXIS_LABEL,
    y_ticks_fontsize: float = FONT_SIZE_AXIS_LABEL,
    y_lim: tuple[float, float] | None = None,
    last_y_label_inside: bool = False,
    spine_width: float = 0.8,
    tick_length: float = 3.5,
    grid: bool = False,
) -> None:

    if not len(point) == len(ci_lower) == len(ci_upper):
        raise ValueError(
            f"The number of groups must be equal for `point` {len(point)}, `ci_lower` {len(ci_lower)}, and `ci_upper` {len(ci_upper)}."
        )

    for group_point, group_ci_lower, group_ci_upper in zip(point, ci_lower, ci_upper):
        if not len(group_point) == len(group_ci_lower) == len(group_ci_upper):
            raise ValueError(
                f"The number of values must be equal for each group in `point`, `ci_lower`, and `ci_upper`."
            )
        for p, ci_l, ci_u in zip(group_point, group_ci_lower, group_ci_upper):
            if any(v < 0 for v in [p, ci_l, ci_u]):
                raise ValueError("All values must be non-negative.")
            if not ci_l <= p <= ci_u:
                raise ValueError(f"Point ({p}) is not within the confidence interval ([{ci_l}, {ci_u}]).")

    n_groups = len(point)
    n_values = sum(len(group) for group in point)

    if len(names) != n_values:
        raise ValueError(f"The length of `names` ({len(names)}) must be equal to the number of values ({n_values}).")

    if colors is None:
        colors_ = QualColors().get_n(n=n_groups)
    elif len(colors) != n_groups:
        raise ValueError(
            f"If `colors` is provided, its length ({len(colors)}) must be equal to the number of groups ({n_groups})."
        )
    else:
        colors_ = colors

    x = calculate_x_group(y=point, spacing=spacing, group_spacing=group_spacing)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title(
        title,
        fontsize=FONT_SIZE_TITLE,
        y=Y_TITLE,
    )

    ax.tick_params(width=spine_width, length=tick_length)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    if grid:
        ax.yaxis.grid(True, which="major", alpha=0.2, linewidth=spine_width)
        ax.set_axisbelow(True)

    ax.set_xlim(x[0][0] - spacing / 2, x[-1][-1] + spacing / 2)
    ax.set_xticks(list(itertools.chain(*x)))
    ax.set_xticklabels(names, fontsize=names_fontsize, rotation=names_rotation)

    ax.set_ylabel(y_label, fontsize=y_label_fontsize)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(axis="y", labelsize=y_ticks_fontsize)
    if last_y_label_inside:
        ax.get_yticklabels()[-1].set_va("top")  # type: ignore[attr-defined]

    for x_group, point_group, ci_lower_group, ci_upper_group, color_group in zip(x, point, ci_lower, ci_upper, colors_):
        neg_error = [point - lower for point, lower in zip(point_group, ci_lower_group)]
        pos_error = [upper - point for point, upper in zip(point_group, ci_upper_group)]
        ax.errorbar(
            x_group,
            point_group,
            yerr=[neg_error, pos_error],
            color=color_group,
            ecolor=color_group,  # type: ignore[arg-type]
            linestyle="none",
            fmt="o",
            markersize=marker_diameter,
            capsize=capsize,
            capthick=capthick,
            elinewidth=elinewidth,
        )

    save_plot(fig=fig, dir_=dir_, file_name=file_name)


def plot_dots_groups(
    dir_: Path,
    file_name: str,
    names: list[str],
    y: list[list[list[float]]],
    jitter: float = 0.0,
    spacing: float = 0.5,
    group_spacing: float = 1.0,
    marker_diameter: float = 7.0,
    colors: tuple[Color, ...] | None = None,
    colors_are_groups: bool = True,
    fig_size: tuple[float, float] = FIG_SIZE,
    title: str = "",
    names_fontsize: float = FONT_SIZE_AXIS_LABEL,
    names_rotation: int = 0,
    y_label: str = "",
    y_label_fontsize: float = FONT_SIZE_AXIS_LABEL,
    y_ticks_fontsize: float = FONT_SIZE_AXIS_LABEL,
    y_lim: tuple[float, float] | None = None,
    last_y_label_inside: bool = False,
    spine_width: float = 0.8,
    tick_length: float = 3.5,
    grid: bool = False,
    legend: Legend | None = None,
) -> None:
    n_groups = len(y)
    n_values = sum(len(group) for group in y)

    if len(names) != n_values:
        raise ValueError(f"The length of `names` ({len(names)}) must be equal to the number of values ({n_values}).")

    if colors is None:
        if colors_are_groups:
            colors_ = QualColors().get_n(n_groups)
        else:
            colors_ = QualColors().get_n(n_groups)
    else:
        colors_ = colors

    n_colors = len(colors_)

    if colors_are_groups and n_colors != n_groups:
        raise ValueError(
            f"If `colors_are_groups` is True, the length of `colors` ({n_colors}) must be equal to the number of groups ({n_groups})."
        )
    elif max(len(inner) for outer in y for inner in outer) != n_colors:
        raise ValueError(f"If `colors_are_groups` is False, each value must have a corresponding color.")

    x = calculate_x_group(y=y, spacing=spacing, group_spacing=group_spacing)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, y=Y_TITLE)

    ax.tick_params(width=spine_width, length=tick_length)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    if grid:
        ax.yaxis.grid(True, which="major", alpha=0.2, linewidth=spine_width)
        ax.set_axisbelow(True)

    ax.set_xlim(x[0][0] - spacing / 2, x[-1][-1] + spacing / 2)
    ax.set_xticks(list(itertools.chain(*x)))
    ax.set_xticklabels(names, fontsize=names_fontsize, rotation=names_rotation)

    ax.set_ylabel(y_label, fontsize=y_label_fontsize)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(axis="y", labelsize=y_ticks_fontsize)
    if last_y_label_inside:
        ax.get_xticklabels()[-1].set_va("top")  # type: ignore[attr-defined]

    s = diameter_to_area(marker_diameter)
    for i, (x_group, y_group) in enumerate(zip(x, y)):
        for x_, y_ in zip(x_group, y_group):
            if colors_are_groups:
                color = [colors_[i]] * len(y_)
            else:
                color = list(colors_)
            jittered_x = np.random.uniform(low=x_ - jitter, high=x_ + jitter, size=len(y_))
            ax.scatter(jittered_x, np.array(y_), color=color, s=s)

    if legend is not None:
        legend.apply(ax)

    save_plot(fig=fig, dir_=dir_, file_name=file_name)


def plot_color_bar(
    dir_: Path,
    color: SeqColors | None = None,
    fig_size: tuple[float, float] = (1, FIG_SIZE[1]),
) -> None:
    if color is None:
        color = SeqColors()

    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=color)
    mappable.set_array([])

    # IMPORTANT: use subplots (tight_layout compatible)
    fig, cax = plt.subplots(figsize=fig_size)

    cbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation="vertical",
        ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    cbar.ax.tick_params(labelsize=12)

    save_plot(fig=fig, dir_=dir_, file_name="color_bar.png")
