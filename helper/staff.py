from typing import Callable

from helper import model_
from helper.print import print_end, print_info
from model.cross_validation import CrossValidation
from model.experiment import Experiment

_MODELS = ["BrT", "Inceptron", "ResNet4D"]


def func_for_all_finished_exps(func: Callable[[Experiment], None]) -> None:
    for model in _MODELS:
        print_info(f"Running function with all finished experiments for `{model}`...")
        finished_experiment_paths = model_.get_finished_experiment_paths(model_name=model)

        for finished_experiment_path in finished_experiment_paths:
            print(finished_experiment_path.name)
            exp = Experiment.load(finished_experiment_path)
            func(exp)

    print_end("Done")


def func_for_runs_of_all_finished_cvs(func: Callable[[Experiment], None]) -> None:
    for model in _MODELS:
        print_info(f"Running function with runs of all finished cross validations for `{model}`...")
        finished_cv_paths = model_.get_finished_cross_validation_paths(model_name=model)

        for finished_cv_path in finished_cv_paths:
            print(finished_cv_path.name, end="")
            cv = CrossValidation.load(finished_cv_path)

            finished_iterations = []

            try:
                for run in cv.runs:
                    func(run)
                    finished_iterations.append(str(run.get_iteration()))

            finally:
                print(f" ({','.join(finished_iterations)})")

    print()
    print_end("Done")


def func_for_all_finished_cvs(func: Callable[[CrossValidation], None]) -> None:
    for model in _MODELS:
        print_info(f"Running function with all finished cross validations for `{model}`...")
        finished_cv_paths = model_.get_finished_cross_validation_paths(model_name=model)

        for finished_cv_path in finished_cv_paths:
            print(finished_cv_path.name)
            cv = CrossValidation.load(finished_cv_path)
            func(cv)

    print()
    print_end("Done")
