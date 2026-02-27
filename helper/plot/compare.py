from itertools import chain
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config.file_names import (
    ACCURACY_COMPARISON_FILE_NAME,
    LOG_LOSS_COMPARISON_FILE_NAME,
    LOSS_COMPARISON_FILE_NAME,
)
from helper import model_, path
from helper.plot.color import QualColors
from helper.plot.main import (
    FIG_SIZE,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TITLE,
    LEGEND_LOC,
    LINE_WIDTH,
    Y_TITLE,
    calculate_scatter_size,
    get_title,
    save_plot,
    set_axis_epoch,
)
from model.experiment import Experiment

# loss | log loss | acc
_COLOR_ALTERNATIVE: QualColors | None = QualColors(
    cmap_name="tab20b", order=[0, 4, 8, 12, 16, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19]
)
_COMP_LOSS_DATASETS: list[Literal["train", "val", "test"]] = ["train", "val", "test"]  # ["train", "val", "test"]
_COMP_ACC_DATASETS: list[Literal["val", "test"]] = ["val", "test"]  # ["val", "test"]
_COMP_PLOT_TITLE: bool = True  # True
_COMP_PLOT_TITLE_TEXT: str = ""  # ""
_COMP_LEGEND_LOC: list[LEGEND_LOC] = [
    "lower right",
    "outside",
    "lower right",
]  # ["lower right", "outside", "lower right"]
_COMP_N_COL: list[int] = [3, 2, 3]  # [3, 2, 3]
_COMP_EXP_NAMES: list[str] | None = None  # None


# loss | log loss | acc
def plot_exp_comparison(
    exps: list[Path] | list[Experiment],
    comp_name: str,
    dir_: Path,
    color_alternative: QualColors | None = _COLOR_ALTERNATIVE,
    comp_loss_datasets: list[Literal["train", "val", "test"]] = _COMP_LOSS_DATASETS,
    comp_acc_datasets: list[Literal["val", "test"]] = _COMP_ACC_DATASETS,
    comp_plot_title: bool = _COMP_PLOT_TITLE,
    comp_plot_title_text: str = _COMP_PLOT_TITLE_TEXT,
    comp_legend_loc: list[LEGEND_LOC] = _COMP_LEGEND_LOC,
    comp_n_col: list[int] = _COMP_N_COL,
    comp_exp_names: list[str] | None = _COMP_EXP_NAMES,
) -> None:
    n_exps = len(exps)
    n_colors = len(QualColors())
    n_colors_alternative = len(color_alternative) if color_alternative else 0
    if n_exps > n_colors and n_colors_alternative > n_colors:
        colors = color_alternative
    else:
        colors = None

    line_width = 1.0 if n_exps >= 8 else LINE_WIDTH

    plot_loss_comparison_from_exps(
        exps=exps,
        comp_name=comp_name,
        dir_=dir_,
        datasets=comp_loss_datasets,
        exp_names=comp_exp_names,
        title=comp_plot_title,
        title_text=comp_plot_title_text,
        colors=colors,
        line_width=line_width,
        legend_loc=comp_legend_loc[0],
        n_col=comp_n_col[0],
    )

    plot_loss_comparison_from_exps(
        exps=exps,
        comp_name=comp_name,
        dir_=dir_,
        datasets=comp_loss_datasets,
        exp_names=comp_exp_names,
        title=comp_plot_title,
        title_text=comp_plot_title_text,
        colors=colors,
        line_width=line_width,
        logarithmic=True,
        legend_loc=comp_legend_loc[1],
        n_col=comp_n_col[1],
    )

    plot_accuracy_comparison_from_exps(
        exps=exps,
        comp_name=comp_name,
        dir_=dir_,
        datasets=comp_acc_datasets,
        exp_names=comp_exp_names,
        title=comp_plot_title,
        title_text=comp_plot_title_text,
        colors=colors,
        line_width=line_width,
        legend_loc=comp_legend_loc[2],
        n_col=comp_n_col[2],
    )


def plot_run_comparison(
    exp_name: str,
    model_name: str | None = None,
    color_alternative: QualColors | None = _COLOR_ALTERNATIVE,
    comp_loss_datasets: list[Literal["train", "val", "test"]] = _COMP_LOSS_DATASETS,
    comp_acc_datasets: list[Literal["val", "test"]] = _COMP_ACC_DATASETS,
    comp_plot_title: bool = _COMP_PLOT_TITLE,
    comp_plot_title_text: str = _COMP_PLOT_TITLE_TEXT,
    comp_legend_loc: list[LEGEND_LOC] = _COMP_LEGEND_LOC,
    comp_n_col: list[int] = _COMP_N_COL,
) -> None:
    if not model_.get_defined_experiment_names(model_name=model_name, pattern=rf"^{exp_name}$"):
        raise ValueError(f"`{exp_name}` is not a valid experiment name.")

    exps_dir = model_.get_finished_experiment_paths(model_name=model_name, pattern=rf"^{exp_name}\.\d+$")
    exps = [Experiment.load(exp_dir) for exp_dir in exps_dir]
    finished_exps = [exp for exp in exps if exp.is_finished()]
    exp_names = [f"run {exp.get_iteration() + 1}" for exp in finished_exps]

    if len(finished_exps) < 2:
        return

    plot_exp_comparison(
        exps=finished_exps,
        comp_name=exp_name,
        dir_=path.experiment(model_name=model_name, absolute=True) / "plot" / "run" / exp_name,
        color_alternative=color_alternative,
        comp_loss_datasets=comp_loss_datasets,
        comp_acc_datasets=comp_acc_datasets,
        comp_plot_title=comp_plot_title,
        comp_plot_title_text=comp_plot_title_text,
        comp_legend_loc=comp_legend_loc,
        comp_n_col=comp_n_col,
        comp_exp_names=exp_names,
    )


def plot_loss_comparison(
    comp_name: str,
    dir_: Path,
    exp_names: list[str],
    train_loss: list[list[float]] | None,
    val_loss: list[list[float]] | None,
    test_loss: list[float] | None,
    title: bool = False,
    title_text: str = "",
    colors: QualColors | None = None,
    line_width: float = LINE_WIDTH,
    warmup_loss: list[tuple[list[float], float] | None] | None = None,
    plot_min: bool = True,
    logarithmic: bool = False,
    y_lim: tuple[float, float] | None = None,
    legend_loc: LEGEND_LOC = "best",
    n_col: int = 3,
) -> None:
    if train_loss is None and val_loss is None:
        raise ValueError("Either `train_loss` or `val_loss` must not be None.")

    n_exp = len(exp_names)

    if train_loss is not None:
        if n_exp != len(train_loss):
            raise ValueError(
                f"The length of `exp_names` ({n_exp}) must be equal to the length of `train_loss` ({len(train_loss)})."
            )
        if any(len(loss) < 2 for loss in train_loss):
            raise ValueError("All training loss curves must have at least two elements.")

    if val_loss is not None:
        if n_exp != len(val_loss):
            raise ValueError(
                f"The length of `exp_names` ({n_exp}) must be equal to the length of `val_loss` ({len(val_loss)})."
            )
        if any(len(loss) < 2 for loss in val_loss):
            raise ValueError("All validation loss curves must have at least two elements.")

    if test_loss is not None:
        if n_exp != len(test_loss):
            raise ValueError(
                f"The length of `exp_names` ({n_exp}) must be equal to the length of `test_loss` ({len(test_loss)})."
            )

    if warmup_loss is not None:
        if train_loss is None:
            raise ValueError("`warmup_loss` must be None if `train_loss` is None.")

        if len(warmup_loss) != len(train_loss):
            raise ValueError(
                f"The lenght of `warmup_loss` ({len(warmup_loss)}) must be equal to length of `train_loss` if `warmup_loss` is not None."
            )
    else:
        warmup_loss = [] if train_loss is None else [None] * len(train_loss)

    if colors is None:
        colors = QualColors()

    colors_ = colors.get_n(n=n_exp)

    max_n_epochs = max((len(loss) for loss in chain(train_loss or [], val_loss or [])), default=0)
    max_n_warmup_epochs = max((len(w_l[0]) if w_l is not None else 0 for w_l in warmup_loss), default=0)

    plot_title = f"{'Log ' if logarithmic else ''}Loss Comparison" if title else ""
    plot_title += " (train)" if train_loss is not None and val_loss is None and test_loss is None else ""
    plot_title += " (validation)" if val_loss is not None and train_loss is None and test_loss is None else ""

    title_ = get_title(plot_title=plot_title, title_text=title_text)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(
        title_,
        fontsize=FONT_SIZE_TITLE,
        y=Y_TITLE,
    )

    ax.set_xlabel(xlabel="Epoch", fontsize=FONT_SIZE_AXIS_LABEL)

    shift_left_xlim = 1 - min([w_l[1] for w_l in warmup_loss if w_l is not None], default=1)
    x_range = set_axis_epoch(
        ax=ax, n_epochs=max_n_epochs, n_warmup_epochs=max_n_warmup_epochs, shift_left_xlim=shift_left_xlim
    )

    ax.set_ylabel(ylabel=f"{'Log ' if logarithmic else ''}Loss", fontsize=FONT_SIZE_AXIS_LABEL)
    if logarithmic:
        ax.set_yscale("log")

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if any(w_l is not None for w_l in warmup_loss):
        ax.axvline(max_n_warmup_epochs, color="gray", linewidth=0.5)

    if train_loss is not None:
        for warmup_loss_, loss, color in zip(warmup_loss, train_loss, colors_):
            if warmup_loss_ is not None:
                warmup_epochs_diff = max_n_warmup_epochs - len(warmup_loss_[0])
                loss_x = [x_ + 1 - warmup_loss_[1] for x_ in x_range[warmup_epochs_diff : max_n_warmup_epochs - 1]]
                loss_x += list(x_range[max_n_warmup_epochs - 1 : max_n_warmup_epochs + len(loss)])
                loss_y = warmup_loss_[0] + loss
            else:
                loss_x = list(x_range[max_n_warmup_epochs : max_n_warmup_epochs + len(loss)])
                loss_y = loss

            ax.plot(
                loss_x,
                loss_y,
                color=color,
                linewidth=line_width,
                linestyle="--" if val_loss else "-",
                zorder=0,
            )

            if plot_min:
                min_loss_idx = np.argmin(loss)
                ax.scatter(
                    min_loss_idx + x_range[max_n_warmup_epochs],
                    loss[min_loss_idx],
                    color=color,
                    s=calculate_scatter_size(line_width=line_width),
                    zorder=1,
                )

    if val_loss is not None:
        for loss, color in zip(val_loss, colors_):
            ax.plot(
                x_range[max_n_warmup_epochs : max_n_warmup_epochs + len(loss)],
                loss,
                color=color,
                linewidth=line_width,
                zorder=2,
            )

            if plot_min:
                min_loss_idx = np.argmin(loss)
                ax.scatter(
                    min_loss_idx + x_range[max_n_warmup_epochs],
                    loss[min_loss_idx],
                    color=color,
                    s=calculate_scatter_size(line_width=line_width),
                    zorder=3,
                )

    if test_loss is not None:
        for test_loss_, color in zip(test_loss, colors_):
            ax.annotate(
                "▶",
                xycoords=("axes fraction", "data"),
                xy=(1.0, test_loss_),
                ha="right",
                fontsize=12,
                color=color,
                zorder=4,
            )

    handles: list[Line2D | Patch] = []
    labels = []

    if [train_loss, val_loss, test_loss].count(None) < 2:
        if train_loss is not None:
            handles.append(Line2D([], [], color="gray", linestyle="--" if val_loss else "-"))
            labels.append("train loss")

        if val_loss is not None:
            handles.append(Line2D([], [], color="gray"))
            labels.append("val loss")

        if test_loss is not None:
            handles.append(Line2D([], [], marker=">", color="gray", linestyle="", markersize=8))
            labels.append("test loss")

        for color in colors_:
            handles.append(Patch(facecolor=color))  # type: ignore[arg-type]
    else:
        for color in colors_:
            handles.append(Line2D([], [], color=color))  # type: ignore[arg-type]

    for exp_name in exp_names:
        labels.append(exp_name)

    font_size = FONT_SIZE_LEGEND if len(exp_names) < 6 else FONT_SIZE_LEGEND - 2

    if legend_loc == "outside":
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=n_col,
            fontsize=font_size,
        )
    else:
        ax.legend(
            handles=handles,
            labels=labels,
            loc=legend_loc,
            ncol=n_col,
            fontsize=font_size,
        )

    file_name = (
        LOG_LOSS_COMPARISON_FILE_NAME.format(name=comp_name)
        if logarithmic
        else LOSS_COMPARISON_FILE_NAME.format(name=comp_name)
    )
    save_plot(fig=fig, dir_=dir_, file_name=file_name)


def plot_accuracy_comparison(
    comp_name: str,
    dir_: Path,
    exp_names: list[str],
    val_acc: list[list[float]],
    test_acc: list[float] | None,
    title: bool = False,
    title_text: str = "",
    colors: QualColors | None = None,
    line_width: float = LINE_WIDTH,
    plot_min: bool = True,
    y_lim: tuple[float, float] = (0, 1.05),
    legend_loc: LEGEND_LOC = "best",
    n_col: int = 3,
) -> None:
    n_exp = len(exp_names)

    if n_exp != len(val_acc):
        raise ValueError(
            f"The length of `exp_names` ({n_exp}) must be equal to the length of `val_acc` ({len(val_acc)})."
        )

    if test_acc is not None and n_exp != len(test_acc):
        raise ValueError(
            f"The length of `exp_names` ({n_exp}) must be equal to the length of `test_acc` ({len(test_acc)})."
        )

    if any(len(acc) < 2 for acc in val_acc):
        raise ValueError("All validation accuracy curves must have at least two elements.")

    if colors is None:
        colors = QualColors()

    colors_ = colors.get_n(n=n_exp)

    max_n_epochs = max(len(acc) for acc in val_acc)

    plot_title = "Accuracy Comparison" if title else ""
    plot_title += " (validation)" if test_acc is None else ""
    title_ = get_title(plot_title=plot_title, title_text=title_text)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title_, fontsize=FONT_SIZE_TITLE, y=Y_TITLE)

    ax.set_xlabel(xlabel="Epoch", fontsize=FONT_SIZE_AXIS_LABEL)
    set_axis_epoch(ax, max_n_epochs)

    ax.set_ylabel(ylabel="Accuracy", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(y_lim)

    for acc, color in zip(val_acc, colors_):
        ax.plot(np.arange(1, len(acc) + 1), acc, color=color, linewidth=line_width, zorder=0)

        if plot_min:
            max_acc_idx = np.argmax(acc)
            ax.scatter(
                max_acc_idx + 1,
                acc[max_acc_idx],
                color=color,
                s=calculate_scatter_size(line_width=line_width),
                zorder=1,
            )

    if test_acc is not None:
        for test_acc_, color in zip(test_acc, colors_):
            ax.annotate(
                "▶",
                xycoords=("axes fraction", "data"),
                xy=(1, test_acc_),
                ha="right",
                fontsize=12,
                color=color,
                zorder=2,
            )

    handles: list[Line2D | Patch] = []
    labels = []

    if test_acc is not None:
        handles.extend(
            [Line2D([], [], color="gray"), Line2D([], [], marker=">", color="gray", linestyle="", markersize=8)]
        )
        labels.extend(["val accuracy", "test accuracy"])

        for color in colors_:
            handles.append(Patch(facecolor=color))  # type: ignore[arg-type]
    else:
        for color in colors_:
            handles.append(Line2D([], [], color=color))  # type: ignore[arg-type]

    for exp_name in exp_names:
        labels.append(exp_name)

    font_size = FONT_SIZE_LEGEND if len(exp_names) < 6 else FONT_SIZE_LEGEND - 2

    if legend_loc == "outside":
        ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=n_col,
            fontsize=font_size,
        )
    else:
        ax.legend(
            handles=handles,
            labels=labels,
            loc=legend_loc,
            ncol=n_col,
            fontsize=font_size,
        )

    file_name = ACCURACY_COMPARISON_FILE_NAME.format(name=comp_name)
    save_plot(fig=fig, dir_=dir_, file_name=file_name)


def plot_loss_comparison_from_exps(
    exps: list[Experiment] | list[Path],
    comp_name: str,
    dir_: Path,
    datasets: list[Literal["train", "val", "test"]] = ["train", "val", "test"],
    exp_names: list[str] | None = None,
    title: bool = False,
    title_text: str = "",
    colors: QualColors | None = None,
    line_width: float = LINE_WIDTH,
    warmup_loss: bool = True,
    plot_min: bool = True,
    logarithmic: bool = False,
    y_lim: tuple[float, float] | None = None,
    legend_loc: LEGEND_LOC = "best",
    n_col: int = 3,
) -> None:
    from model.experiment import Experiment

    exps_ = [exp if isinstance(exp, Experiment) else Experiment.load(exp) for exp in exps]
    exp_names_ = exp_names if exp_names else [exp.get_name() for exp in exps_]

    loss_train: list[list[float]] = []
    loss_val: list[list[float]] = []
    loss_test: list[float] = []

    if warmup_loss:
        warmup_loss_ = [exp.get_warmup_loss_as_epochs() if exp.has_warmup() else None for exp in exps_]
    else:
        warmup_loss_ = None

    for exp in exps_:
        if "train" in datasets:
            loss_train.append(cast(list[float], exp.get_loss("train")))
        if "val" in datasets:
            loss_val.append(cast(list[float], exp.get_loss("val")))
        if "test" in datasets:
            loss_test.append(cast(float, exp.get_loss("test")))

    plot_loss_comparison(
        comp_name=comp_name,
        dir_=dir_,
        exp_names=exp_names_,
        train_loss=loss_train if loss_train else None,
        val_loss=loss_val if loss_val else None,
        test_loss=loss_test if loss_test else None,
        title=title,
        title_text=title_text,
        colors=colors,
        line_width=line_width,
        warmup_loss=warmup_loss_,
        plot_min=plot_min,
        logarithmic=logarithmic,
        y_lim=y_lim,
        legend_loc=legend_loc,
        n_col=n_col,
    )


def plot_accuracy_comparison_from_exps(
    exps: list[Experiment] | list[Path],
    comp_name: str,
    dir_: Path,
    datasets: list[Literal["val", "test"]],
    exp_names: list[str] | None = None,
    title: bool = False,
    title_text: str = "",
    colors: QualColors | None = None,
    line_width: float = LINE_WIDTH,
    plot_min: bool = True,
    y_lim: tuple[float, float] = (0, 1.05),
    legend_loc: LEGEND_LOC = "best",
    n_col: int = 3,
) -> None:
    from model.experiment import Experiment

    if "val" not in datasets:
        raise ValueError("`val` must be in `datasets`.")

    exps_ = [exp if isinstance(exp, Experiment) else Experiment.load(exp) for exp in exps]
    exp_names_ = exp_names if exp_names else [exp.get_name() for exp in exps_]

    val_acc: list[list[float]] = []
    test_acc: list[float] = []

    for exp in exps_:
        val_acc.append(cast(list[float], exp.get_accuracy("val")))

        if "test" in datasets:
            test_acc.append(cast(float, exp.get_accuracy("test")))

    plot_accuracy_comparison(
        comp_name=comp_name,
        dir_=dir_,
        exp_names=exp_names_,
        val_acc=val_acc,
        test_acc=test_acc if test_acc else None,
        title=title,
        title_text=title_text,
        colors=colors,
        line_width=line_width,
        plot_min=plot_min,
        y_lim=y_lim,
        legend_loc=legend_loc,
        n_col=n_col,
    )
