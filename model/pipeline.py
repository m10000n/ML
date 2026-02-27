import sys
from pathlib import Path
from time import sleep

from config import file_names
from helper import env, path, process, system, time, tmux
from helper.exception import NanError
from helper.plot import cross_validation, experiment
from helper.print import print_end, print_error, print_info, print_start
from helper.process import PROCESS_LOCK
from model import test, train
from model.cross_validation import CrossValidation
from model.experiment import Experiment, ExperimentConfig


def __run_experiment(config: ExperimentConfig, track_time: bool) -> None:
    exp = Experiment(config=config, track_time=track_time)
    tmux.run(
        window_name=exp.get_name().replace(".", "R"), func=_train_test_plot, attach=True, exp_path=str(exp.get_dir())
    )


def __run_cross_validation(config: ExperimentConfig, track_time: bool) -> None:
    cv = CrossValidation(exp_config=config, track_time=track_time)
    tmux.run(window_name=f"{cv.name.replace(".", "CV")}", func=_cross_validation, attach=True, dir_=str(cv.dir_))


def __continue_cross_validation(cv_name: str) -> None:
    dir_ = path.cross_validation() / cv_name

    if not dir_.exists():
        print_error(
            text=f"Failed to continue cross validation. Cross validation with path `{dir_}` not found.", mode="primary"
        )
        sys.exit(1)

    cv = CrossValidation.load(dir_)

    finished_runs = [run for run in cv.runs if run.is_finished()]
    unfinished_runs = [run for run in cv.runs if not run.is_finished()]

    if unfinished_runs:
        print_info(
            f"Found `{len(finished_runs)}` finished run{'s' if len(finished_runs) > 1 else ''} "
            f"and `{len(unfinished_runs)}` unfinished run{'s' if len(unfinished_runs) > 1 else ''} "
            f"for cross validation `{cv_name}`.",
            mode="primary",
        )
    else:
        print_error(f"No unfinished runs found for cross validation `{cv_name}`.", mode="primary")
        sys.exit(1)

    for run in unfinished_runs:
        run.reset()

        run_dir = run.get_dir()
        run_name = run.get_name()
        (run_dir / file_names.DEPENDENCY_FILE_NAME.format(exp_name=run_name)).unlink(missing_ok=True)
        (run_dir / file_names.TRAINED_MODEL_FILE_NAME.format(exp_name=run_name)).unlink(missing_ok=True)

    tmux.run(window_name=f"{cv_name.replace(".", "CV")}", func=_cross_validation, attach=True, dir_=str(dir_))


def _cross_validation(dir_: Path) -> None:
    cv = CrossValidation.load(dir_)

    id_run: list[tuple[str, Experiment]] = []

    finished_runs = [run for run in cv.runs if run.is_finished()]
    unfinished_runs = [run for run in cv.runs if not run.is_finished()]

    tmux_name = cv.name.replace(".", "CV") + "R{run_iteration}"

    try:
        mode_str = "Continue" if finished_runs else "Start"
        pu = system.get_pu()
        pu_str = ",".join([str(gpu_idx) for gpu_idx in pu]) if pu != "cpu" else "CPU"
        print_start(
            text=f"{mode_str} Cross Validation | {time.now_str()} | Experiment: {cv.name} | PU: {pu_str} | Number of Workers: {system.get_num_workers()}.",
            mode="primary",
        )

        for run in finished_runs:
            print_info(f"Run #{run.get_iteration()} already finished.")

        for run in unfinished_runs:
            print_start(f"Start run #{run.get_iteration()}.")

            id_ = process.get_new_id()
            id_run.append((id_, run))

            tmux.run(
                window_name=tmux_name.format(run_iteration=run.get_iteration()),
                func=_train_test_plot,
                process_id=id_,
                exp_path=str(run.get_dir()),
                plot=False,
            )

            while not process.is_tracked(id_, with_lock=False):
                sleep(0.1)

            while process.is_tracked(id_, with_lock=False):
                sleep(10)

            print_end(f"Finished run #{run.get_iteration()}.")

        if not cv.all_runs_finished(reload=True):
            print_error(f"Not all runs finished. Skipping evaluation and plotting.")
            print_error(f"Finished Cross Validation.", mode="primary")
        else:
            print_start("Start evaluating results.")
            cv.evaluate()
            print_end("Finished evaluating results.")

            cross_validation.plot_cross_validation(cv)

            print_end(text="Finished Cross Validation.", mode="primary")

    except KeyboardInterrupt:
        for id_, run in id_run:
            with PROCESS_LOCK:
                if process.is_tracked(id_, with_lock=False):
                    process_ = process.get(id_, with_lock=False)
                    process_.kill(with_lock=False)

                    tmux_name = next(meta for meta in (process_.meta or []) if meta.startswith("tmux_window: "))
                    tmux.kill_window(tmux_name[len("tmux_window: ") :])

                    run.reload()
                    run.set_status("aborted")
                    run.set_aborted_reason("keyboard_interrupt")
                    run.write()

        print_info("Aborted by user.")


def _train_test_plot(exp_path: str, plot: bool = True) -> None:
    exp = Experiment.load(exp_path)
    exp_pu = exp.get_pu()

    try:
        system.pu_sanity_check(exp_pu)
    except RuntimeError as e:
        print_error(f"Failed to run the experiment. {e.args[0]}")
        sys.exit(1)

    system.set_pu(exp_pu)
    system.set_num_workers(exp.get_num_workers())
    system.set_prefetch_factor(exp.get_prefetch_factor())

    try:
        system.autocast_sanity_check(exp.with_autocast())
    except RuntimeError as e:
        print_error(f"Failed to run the experiment. {e.args[0]}")
        sys.exit(1)

    system.set_autocast(exp.with_autocast())

    env.save_dependencies(path=exp.get_dir() / file_names.DEPENDENCY_FILE_NAME.format(exp_name=exp.get_name()))

    exp.set_status("running")
    exp.write()
    exp.download_datasets()

    try:

        print()

        train.main(exp)

        print()

        test.main(exp)

        print()

        if exp.get_status() == "running":
            exp.set_status("done")
            exp.write()

        print_start("Start evaluating results.")
        exp.evaluate(strict=False)
        print_end("Finished evaluating results.")

        if plot:
            experiment.plot_experiment(exp=exp)

    except BaseException as e:
        exp.set_status("aborted")

        if isinstance(e, KeyboardInterrupt):
            exp.set_aborted_reason("keyboard_interrupt")
            print_info("Experiment aborted by user.")
        elif isinstance(e, NanError):
            print_error(f"Experiment aborted due to NaN error. {e.args[0]}")
            exp.set_aborted_reason("nan")
        else:
            exp.set_aborted_reason("other")
            raise
    finally:
        exp.write()
        if exp.is_finished():
            exp.create_overview()
