from __future__ import annotations

from typing import cast

from config.file_names import PLOT_CREATION_LOG_FILE_NAME
from helper import file, path
from helper.plot.compare import plot_exp_comparison
from helper.plot.experiment import plot_confusion_matrix
from helper.print import print_end, print_error, print_start
from model.cross_validation import CrossValidation, CrossValidationResult


def plot_cross_validation(cv: CrossValidation, title: bool = False, title_text: str = "") -> None:
    print_start(text="Start plotting.", mode="primary")

    if not cv.is_evaluated():
        raise ValueError("The cross validation results are not available. Call `evaluate` first.")

    error_msg = []

    runs_no_epoch_names = [run.get_name() for run in cv.runs if run.get_total_epochs() < 2]
    runs_no_actual_names = [run.get_name() for run in cv.runs if len(run.get_confusion_ids()) == 0]

    if runs_no_epoch_names:
        error_msg.append(
            f"No epochs found for the following run{'' if len(runs_no_epoch_names) == 1 else 's'}: {", ".join(runs_no_epoch_names)}."
        )

    if runs_no_actual_names:
        error_msg.append(
            f"No test data found for the following run{'' if len(runs_no_actual_names) == 1 else 's'}: {", ".join(runs_no_actual_names)}."
        )

    plot_dir = cv.get_plot_dir()
    log_file_path = plot_dir / PLOT_CREATION_LOG_FILE_NAME.format(exp_name=cv.name)

    if not error_msg:
        try:
            plot_confusion_matrix(
                exp_name=cv.name,
                dir_=plot_dir,
                actual=cast(list[int], cast(CrossValidationResult, cv.results).confusion["actual"]),
                predicted=cast(list[int], cast(CrossValidationResult, cv.results).confusion["predicted"]),
                class_names=cv.class_names,
                title=title,
                title_text=title_text,
            )
        except BaseException as e:
            error_msg.append(f"Failed to plot confusion matrix. {e.args[0]}")

        try:
            plot_exp_comparison(
                exps=cv.runs,
                comp_name=cv.name,
                dir_=plot_dir,
                color_alternative=None,
                comp_loss_datasets=["train", "val", "test"],
                comp_acc_datasets=["val", "test"],
                comp_plot_title=title,
                comp_plot_title_text=title_text,
                comp_legend_loc=["upper right", "outside", "lower right"],
                comp_n_col=[1, 2, 1],
                comp_exp_names=[f"run {run.get_iteration() + 1}" for run in cv.runs],
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
