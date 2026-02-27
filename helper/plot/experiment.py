from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config.file_names import (
    CONFUSION_PLOT_FILE_NAME,
    LOG_LOSS_ACCURACY_PLOT_FILE_NAME,
    LOSS_ACCURACY_PLOT_FILE_NAME,
    PLOT_CREATION_LOG_FILE_NAME,
)
from helper import file, model_, path, statistic
from helper.helper_ import round_to_str
from helper.plot.color import BLACK, WHITE, Color, QualColors, SeqColors
from helper.plot.compare import plot_run_comparison
from helper.plot.main import (
    FIG_SIZE,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TITLE,
    LINE_WIDTH,
    Y_TITLE,
    get_title,
    save_plot,
    set_axis_epoch,
)
from helper.print import print_end, print_error, print_start

if TYPE_CHECKING:
    from model.experiment import Experiment


def plot_experiment(exp: Experiment, title: bool = False, title_text: str = "") -> None:
    print_start(text="Start plotting.", mode="primary")

    if not exp.is_evaluated():
        raise ValueError("The experiment results are not available. Call `evaluate` first.")

    error_msg = []

    if exp.get_total_epochs() < 2:
        error_msg.append(f"No epochs found.")

    if len(exp.get_confusion_ids()) == 0:
        error_msg.append(f"No test data found.")

    log_file_path = exp.get_plot_dir() / PLOT_CREATION_LOG_FILE_NAME.format(exp_name=exp.get_name())

    if not error_msg:
        try:
            plot_confusion_matrix_from_exp(exp=exp, title=title, title_text=title_text)
        except BaseException as e:
            error_msg.append(f"Failed to plot confusion matrix. {e.args[0]}")

        try:
            plot_loss_accuracy_from_exp(exp=exp, title=title, title_text=title_text)
        except BaseException as e:
            error_msg.append(f"Failed to plot loss and accuracy. {e.args[0]}")

        try:
            plot_loss_accuracy_from_exp(exp=exp, logarithmic=True, title=title, title_text=title_text)
        except BaseException as e:
            error_msg.append(f"Failed to plot loss and accuracy (logarithmic). {e.args[0]}")

        try:
            exp_name = exp.get_name().split(".")[0]
            finished_exp_paths = model_.get_finished_experiment_paths(pattern=rf"^{exp_name}\.\d+$")

            if len(finished_exp_paths) > 1:
                plot_run_comparison(
                    exp_name=exp_name,
                    color_alternative=None,
                    comp_loss_datasets=["train", "val", "test"],
                    comp_acc_datasets=["val", "test"],
                    comp_plot_title=title,
                    comp_plot_title_text=title_text,
                    comp_legend_loc=["upper right", "outside", "lower right"],
                    comp_n_col=[1, 2, 1],
                )
        except BaseException as e:
            error_msg.append(f"Failed to plot run comparison. {e.args[0]}")

    if error_msg:
        file.write_lines(path=log_file_path, lines=error_msg, overwrite=True)
        print_error(
            text=f"Something went wrong while creating plots. Please check the log file at `{path.make_relative(log_file_path)}`.",
            mode="primary",
        )
    else:
        if log_file_path.exists():
            log_file_path.unlink()
        print_end(text="Finished plotting.", mode="primary")


def plot_loss_accuracy(
    exp_name: str,
    dir_: Path,
    loss: dict[Literal["train", "val", "test"], list[float] | float],
    accuracy: dict[Literal["val", "test"], list[float] | float],
    title: bool = False,
    title_text: str = "",
    colors: (
        dict[
            Literal["train_loss", "val_loss", "test_loss", "val_accuracy", "test_accuracy"],
            Color,
        ]
        | None
    ) = None,
    linestyle: bool = True,
    warmup_loss: tuple[list[float], float] | None = None,
    logarithmic: bool = False,
) -> None:
    train_loss = cast(list[float], loss["train"])
    val_loss = cast(list[float], loss["val"])
    test_loss = cast(float, loss["test"])
    val_acc = cast(list[float], accuracy["val"])
    test_acc = cast(float, accuracy["test"])

    if len(val_loss) != len(val_acc):
        raise ValueError(
            f"The length of the validation loss ({len(val_loss)}) and validation accuracy ({len(val_acc)}) must be the same."
        )

    n_epochs = len(val_loss)

    if n_epochs < 2:
        raise ValueError(f"The number of epochs ({n_epochs}) must be greater than 1.")

    n_warmup_epochs = 0 if warmup_loss is None else len(warmup_loss[0])

    plot_title = f"{'Log ' if logarithmic else ''}Loss and Accuracy" if title else ""
    title_ = get_title(plot_title=plot_title, title_text=title_text)

    if colors is None:
        colors_ = QualColors().get_n(n=5)
        colors = {
            "train_loss": colors_[0],
            "val_loss": colors_[1],
            "test_loss": colors_[2],
            "val_accuracy": colors_[3],
            "test_accuracy": colors_[4],
        }

    color_train_loss = colors["train_loss"]
    color_val_loss = colors["val_loss"]
    color_test_loss = colors["test_loss"]
    color_val_accuracy = colors["val_accuracy"]
    color_test_accuracy = colors["test_accuracy"]

    linestyle_loss_train, linestyle_loss_val, linestyle_accuracy_val = "-", "-", "-"

    if linestyle:
        linestyle_loss_train = ":"
        linestyle_loss_val = "--"

    # create plot
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    ax2 = ax1.twinx()

    ax1.set_title(
        title_,
        fontsize=FONT_SIZE_TITLE,
        y=Y_TITLE,
    )

    ax1.set_xlabel("Epoch", fontsize=FONT_SIZE_AXIS_LABEL)
    x_range = set_axis_epoch(
        ax=ax1,
        n_epochs=n_epochs,
        n_warmup_epochs=n_warmup_epochs,
        shift_left_xlim=0 if warmup_loss is None else 1 - warmup_loss[1],
    )

    ax1.set_ylabel(f"{'Log ' if logarithmic else ''}Loss", fontsize=FONT_SIZE_AXIS_LABEL)
    if logarithmic:
        ax1.set_yscale("log")
    ax1.spines["left"].set_zorder(10)

    ax2.set_ylabel("Accuracy", fontsize=FONT_SIZE_AXIS_LABEL)
    ax2.set_ylim(0, 1.05)
    ax2.spines["right"].set_zorder(10)

    ## warmup
    if warmup_loss is not None:
        ax1.axvline(n_warmup_epochs, color="gray", linewidth=0.5)

    # loss
    ## warmup
    if warmup_loss is not None:
        loss_x = [x_ + 1 - warmup_loss[1] for x_ in x_range[: n_warmup_epochs - 1]] + list(
            x_range[n_warmup_epochs - 1 :]
        )
        loss_y = warmup_loss[0] + train_loss
    else:
        loss_x = list(x_range[n_warmup_epochs:])
        loss_y = train_loss

    # ## train
    min_train_loss_idx = np.argmin(np.array(train_loss))
    min_train_loss = train_loss[min_train_loss_idx]

    ax1.plot(
        loss_x,
        loss_y,
        label=f"Training Loss (min: {round_to_str(x=min_train_loss, digits=4)})",
        linewidth=LINE_WIDTH,
        color=color_train_loss,
        linestyle=linestyle_loss_train,
        zorder=0,
    )
    ax1.scatter(min_train_loss_idx + x_range[n_warmup_epochs], min_train_loss, color=color_train_loss, zorder=3)

    ## val
    min_val_loss_idx = np.argmin(np.array(val_loss))
    min_val_loss = val_loss[min_val_loss_idx]

    ax1.plot(
        x_range[n_warmup_epochs:],
        val_loss,
        label=f"Validation Loss (min: {round_to_str(x=min_val_loss, digits=4)})",
        linewidth=LINE_WIDTH,
        color=color_val_loss,
        linestyle=linestyle_loss_val,
        zorder=1,
    )
    ax1.scatter(min_val_loss_idx + x_range[n_warmup_epochs], min_val_loss, color=color_val_loss, zorder=4)

    ## test
    ax1.annotate(
        "◀",
        xycoords=("axes fraction", "data"),
        xy=(0, test_loss),
        ha="left",
        fontsize=12,
        color=color_test_loss,
        zorder=6,
    )

    # accuracy
    ## val
    max_val_acc_idx = np.argmax(np.array(val_acc))
    max_val_acc = val_acc[max_val_acc_idx]

    ax2.plot(
        x_range[n_warmup_epochs:],
        val_acc,
        label=f"Validation Accuracy (max: {round_to_str(x=max_val_acc, digits=4)})",
        linewidth=LINE_WIDTH,
        color=color_val_accuracy,
        linestyle=linestyle_accuracy_val,
        zorder=2,
    )
    ax2.scatter(max_val_acc_idx + x_range[n_warmup_epochs], max_val_acc, color=color_val_accuracy, zorder=5)

    ## test
    ax2.annotate(
        "▶",
        xycoords=("axes fraction", "data"),
        xy=(1, test_acc),
        ha="right",
        fontsize=12,
        color=color_test_accuracy,
        zorder=6,
    )

    # legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    test_loss_handle = Line2D([], [], marker="<", color=color_test_loss, linestyle="", markersize=8)  # type: ignore
    test_acc_handle = Line2D([], [], marker=">", color=color_test_accuracy, linestyle="", markersize=8)  # type: ignore

    lines = handles1 + handles2 + [test_loss_handle, test_acc_handle]
    labels = (
        labels1
        + labels2
        + [
            f"Test Loss ({round_to_str(x=test_loss, digits=4)})",
            f"Test Accuracy ({round_to_str(x=test_acc, digits=4)})",
        ]
    )

    plt.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=FONT_SIZE_LEGEND,
    )

    # save
    file_name = (
        LOG_LOSS_ACCURACY_PLOT_FILE_NAME.format(exp_name=exp_name)
        if logarithmic
        else LOSS_ACCURACY_PLOT_FILE_NAME.format(exp_name=exp_name)
    )
    save_plot(fig=fig, dir_=dir_, file_name=file_name)


def plot_confusion_matrix(
    exp_name: str,
    dir_: Path,
    actual: list[int],
    predicted: list[int],
    class_names: list[str],
    title: bool = False,
    title_text: str = "",
    color: SeqColors | None = None,
) -> None:

    if color is None:
        color = SeqColors()

    n_classes = len(class_names)

    plot_title = "Confusion" if title else ""
    title_ = get_title(plot_title=plot_title, title_text=title_text)

    # create plot
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title_, fontsize=FONT_SIZE_TITLE, y=Y_TITLE)
    ax.set_xlabel(xlabel="Predicted Class", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel(ylabel="True Class", fontsize=FONT_SIZE_AXIS_LABEL)
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # confusion matrix
    conf_matrix = statistic.create_normalized_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)

    im = ax.imshow(conf_matrix, interpolation="nearest", cmap=color)
    fig.colorbar(im, ax=ax)

    threshold = conf_matrix.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            value = conf_matrix[i, j]
            color_ = WHITE if value > threshold else BLACK
            ax.text(
                x=j,
                y=i,
                s=f"{round_to_str(x=value, digits=3)}",
                ha="center",
                va="center",
                color=color_,
                fontsize=10,
            )

    # save
    save_plot(fig=fig, dir_=dir_, file_name=CONFUSION_PLOT_FILE_NAME.format(exp_name=exp_name))


def plot_loss_accuracy_from_exp(
    exp: Experiment | Path,
    title: bool = False,
    title_text: str = "",
    colors: (
        dict[
            Literal["train_loss", "val_loss", "test_loss", "val_accuracy", "test_accuracy"],
            Color,
        ]
        | None
    ) = None,
    linestyle: bool = True,
    warmup_loss: bool = True,
    logarithmic: bool = False,
) -> None:
    from model.experiment import Experiment

    exp_ = exp if isinstance(exp, Experiment) else Experiment.load(exp)

    accuracy = exp_.get_accuracy_()

    if accuracy["test"] is None:
        raise ValueError("The test accuracy is not available.")

    accuracy_: dict[Literal["val", "test"], list[float] | float] = {
        "val": cast(list[float], accuracy["val"]),
        "test": cast(float, accuracy["test"]),
    }

    loss: dict[Literal["train", "val", "test"], list[float] | float] = {
        "train": exp_.get_loss("train"),
        "val": exp_.get_loss("val"),
        "test": exp_.get_loss("test"),
    }

    if warmup_loss and exp_.has_warmup():
        warmup_loss_ = exp_.get_warmup_loss_as_epochs()
    else:
        warmup_loss_ = None

    plot_loss_accuracy(
        exp_name=exp_.get_name(),
        dir_=exp_.get_plot_dir(),
        loss=loss,
        accuracy=accuracy_,
        title=title,
        title_text=title_text,
        colors=colors,
        linestyle=linestyle,
        warmup_loss=warmup_loss_,
        logarithmic=logarithmic,
    )


def plot_confusion_matrix_from_exp(
    exp: Experiment | Path, title: bool = False, title_text: str = "", color: SeqColors | None = None
) -> None:
    from model.experiment import Experiment

    exp_ = exp if isinstance(exp, Experiment) else Experiment.load(exp)

    plot_confusion_matrix(
        exp_name=exp_.get_name(),
        dir_=exp_.get_plot_dir(),
        actual=exp_.get_confusion_actual(),
        predicted=exp_.get_confusion_predicted(),
        class_names=exp_.get_classes(),
        title=title,
        title_text=title_text,
        color=color,
    )
