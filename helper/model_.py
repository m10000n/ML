import re
import sys
from pathlib import Path
from re import Pattern
from types import FunctionType
from typing import List, Literal, Tuple

import torch
from torchinfo import summary as summary_

from config import project
from helper import file, function, fvcore, path
from helper.class_ import Model_, ModelConfig_
from helper.print import print_error, print_info

##### config start #####
_SUMMARY_COLUMNS: list[str] = ["input_size", "output_size", "num_params", "kernel_size", "trainable"]
##### config end #####

_MODEL_CONFIG_PATH = "{architecture_path}/model_config.py"
_EXPERIMENT_CONFIG_PATH = "{experiment_path}/experiment_config.py"

_DEFAULT_BATCH_SIZE: int = 1


_EARLY_STOP_DIR = path.tmp(absolute=True) / "experiment" / "early_stop"
_IS_ACTIVE_PATH_ = str(_EARLY_STOP_DIR) + "/{exp_name}"


# model
def __new(name: str) -> None:
    model_dir = path.model(absolute=True) / name

    if model_dir.exists():
        print_error(f"Model `{name}` already exists.")
        sys.exit(1)

    model_dir.mkdir(parents=False, exist_ok=False)
    (model_dir / "__init__.py").touch(exist_ok=False)

    architecture_dir = model_dir / "architecture"
    architecture_dir.mkdir(parents=False, exist_ok=False)
    (architecture_dir / "__init__.py").touch(exist_ok=False)
    (architecture_dir / "model_config.py").touch(exist_ok=False)

    experiment_dir = model_dir / "experiment"
    experiment_dir.mkdir(parents=False, exist_ok=False)
    (experiment_dir / "__init__.py").touch(exist_ok=False)
    (experiment_dir / "experiment_config.py").touch(exist_ok=False)

    print_info(f"Created model structure.")


def __available_models(details: bool = False, pattern: str = "") -> None:
    model_name = project.get_model()

    try:
        functions = _get_model_config_f(model_name=model_name, pattern=pattern)
    except ModuleNotFoundError as e:
        print_error(f"Failed to display available models for `{model_name}`. {e.args[0]}")
        sys.exit(1)

    if functions:
        print(f"Available models for `{model_name}`:")
        for name, func in functions:
            config: ModelConfig_ = func()
            if details:
                print(f"\t{name} - {config.description}")
            else:
                print(f"\t{name}")
    else:
        model_config_path = _MODEL_CONFIG_PATH.format(architecture_path=path.architecture(absolute=False))
        print_info(
            f"No models found for `{model_name}`. Define a model configuration function in `{model_config_path}`."
        )


def __summary(model_name: str, batch_size: int) -> None:
    batch_size_ = batch_size if batch_size is not None else _DEFAULT_BATCH_SIZE

    error_message = f"Failed to display a summary of `{project.get_model()} - {model_name}`. "
    try:
        functions = _get_model_config_f()
    except ModuleNotFoundError as e:
        print_error(f"{error_message}{e.args[0]}")
        sys.exit(1)

    for name, func in functions:
        if name == model_name:
            model_config = func()
            model = Model_.create(model_config).cpu()
            input = torch.randn(size=(batch_size_, *model_config.input_shape), device="cpu")
            summary_(model, input_data=input, col_names=_SUMMARY_COLUMNS)
            return

    print_error(f"{error_message}Model not found.")
    sys.exit(1)


def __flops(model_name: str, batch_size: int) -> None:
    batch_size_ = batch_size if batch_size is not None else _DEFAULT_BATCH_SIZE

    error_message = f"Failed to display FLOPs of `{project.get_model()} - {model_name}`. "
    try:
        functions = _get_model_config_f()
    except ModuleNotFoundError as e:
        print_error(f"{error_message}{e.args[0]}")
        sys.exit(1)

    for name, func in functions:
        if name == model_name:
            model_config = func()
            model = Model_.create(model_config).cpu()
            fvcore.analyze_flops(model=model, input_shape=model_config.input_shape, batch_size=batch_size_)
            return

    print_error(f"{error_message}Model not found.")
    sys.exit(1)


def _get_model_config_f(model_name: str | None = None, pattern: str | Pattern = "") -> List[Tuple[str, FunctionType]]:
    model_config_path = _MODEL_CONFIG_PATH.format(architecture_path=path.architecture(model_name=model_name))

    try:
        return function._get_f(file_path=model_config_path, pattern=pattern)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Model configuration file {model_config_path}) not found.")


# experiment
def __available_experiments(details: bool = False, pattern: str = "") -> None:
    from model import experiment
    from model.experiment import ExperimentConfig

    model_name = project.get_model()
    try:
        functions = _get_experiment_config_f(model_name=model_name, pattern=pattern)
    except ModuleNotFoundError as e:
        print_error(f"Failed to display available experiments for `{model_name}`. {e.args[0]}.")
        sys.exit(1)

    configs: list[ExperimentConfig] = [func() for _, func in functions]

    if configs:
        name_status = [
            (experiment.get_exp_name(exp_path).split(".")[0], experiment.get_status_dir(exp_path))
            for exp_path in get_finished_experiment_paths(model_name=model_name)
        ]

        output = []

        for config in configs:
            info = config.get_info(details)
            status = [status for name, status in name_status if name == config.name_no_iteration]
            info[0] += f" ({', '.join(status)})" if status else ""
            output.append("".join(info))

        print(f"Available experiments for `{project.get_model()}`:")
        if details:
            print("\n\n".join(output))
        else:
            print("\n".join(["\t" + output_ for output_ in output]))
    else:
        exp_config_path = _EXPERIMENT_CONFIG_PATH.format(experiment_path=path.experiment(absolute=False))
        print_info(
            f"No experiments found for `{model_name}`. Define an experiment configuration function in `{exp_config_path}`."
        )


def __run_experiment(exp_name: str, cross_validation: bool, track_time: bool) -> None:
    from model import pipeline

    error_message = f"Failed to run `{project.get_model()} - {exp_name}`. "
    try:
        functions = _get_experiment_config_f()
    except ModuleNotFoundError as e:
        print_error(f"{error_message}{e.args[0]}.")
        sys.exit(1)

    for name, func in functions:
        if name == exp_name:
            config = func()
            if cross_validation:
                pipeline.__run_cross_validation(config=config, track_time=track_time)
            else:
                pipeline.__run_experiment(config=config, track_time=track_time)
            return

    print_error(f"{error_message}Experiment not found.")
    sys.exit(1)


def __experiment_results(pattern: str = "") -> None:
    from model.experiment import Experiment

    model_name = project.get_model()
    try:
        exp_paths = get_finished_experiment_paths(model_name=model_name, pattern=pattern)
    except ModuleNotFoundError as e:
        print_error(f"Failed to display results of experiments for `{model_name}`. {e.args[0]}.")
        sys.exit(1)

    exps = [Experiment.load(exp_path) for exp_path in exp_paths]

    if exps:
        ljust = max(len(name) for name in [exp.get_name() for exp in exps])
        print(f"Results of experiments for `{model_name}`:")
        for exp in exps:
            print(f"\t{exp.get_result_str(ljust=ljust)}")
    else:
        print_info(f"No finished experiments found for `{model_name}`.")


def get_defined_experiment_names(model_name: str | None = None, pattern: str | Pattern = "") -> list[str]:
    return [
        name for name, _ in _get_experiment_config_f(model_name=model_name) if re.search(pattern=pattern, string=name)
    ]


def get_experiment_paths(model_name: str | None = None, pattern: str | Pattern = "") -> list[Path]:
    defined_exps = get_defined_experiment_names(model_name=model_name)
    exp_paths = []

    for path_ in path.get_dirs(path.experiment(model_name=model_name, absolute=True)):
        if path_.name.split(".")[0] in defined_exps and re.search(pattern=pattern, string=path_.name.split("~")[0]):
            exp_paths.append(path_)

    return exp_paths


def get_experiment_names(model_name: str | None = None, pattern: str | Pattern = "") -> list[str]:
    return [exp_path.name.split("~")[0] for exp_path in get_experiment_paths(model_name=model_name, pattern=pattern)]


def get_running_experiment_paths(model_name: str | None = None, pattern: str | Pattern = "") -> list[Path]:
    from model import experiment

    return [
        exp_path
        for exp_path in get_experiment_paths(model_name=model_name, pattern=pattern)
        if experiment.is_running(exp_path)
    ]


def get_running_experiment_names(model_name: str | None = None, pattern: str | Pattern = "") -> list[str]:
    return [
        exp_path.name.split("~")[0] for exp_path in get_running_experiment_paths(model_name=model_name, pattern=pattern)
    ]


def get_finished_experiment_paths(model_name: str | None = None, pattern: str | Pattern = "") -> list[Path]:
    from model import experiment

    return [
        exp_path
        for exp_path in get_experiment_paths(model_name=model_name, pattern=pattern)
        if experiment.is_finished(exp_path)
    ]


def get_finished_experiment_names(model_name: str | None = None, pattern: str | Pattern = "") -> list[str]:
    return [
        exp_path.name.split("~")[0]
        for exp_path in get_finished_experiment_paths(model_name=model_name, pattern=pattern)
    ]


def _get_experiment_config_f(
    model_name: str | None = None, pattern: str | Pattern = ""
) -> List[Tuple[str, FunctionType]]:
    experiment_config_path = _EXPERIMENT_CONFIG_PATH.format(experiment_path=path.experiment(model_name=model_name))

    try:
        return function._get_f(file_path=experiment_config_path, pattern=pattern)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Experiment configuration file ({experiment_config_path}) not found.")


# cross validation
def __continue_cross_validation(cv_name: str) -> None:
    from model import pipeline

    pipeline.__continue_cross_validation(cv_name)


def get_cross_validation_paths(model_name: str | None = None, pattern: str | Pattern = "") -> list[Path]:
    defined_exps = get_defined_experiment_names(model_name=model_name)
    cv_paths = []

    cv_path = path.cross_validation(model_name=model_name, absolute=True)

    if not cv_path.exists():
        return []

    for path_ in path.get_dirs(cv_path):
        path_name = path_.name.split(".")[0]
        if path_name in defined_exps and re.search(pattern=pattern, string=path_name):
            cv_paths.append(path_)

    return cv_paths


def get_cross_validation_names(model_name: str | None = None, pattern: str | Pattern = "") -> list[str]:
    return [path_.name for path_ in get_cross_validation_paths(model_name=model_name, pattern=pattern)]


def get_running_cross_validation_paths(model_name: str | None = None, pattern: str | Pattern = "") -> list[Path]:
    return _get_status_cross_validation_paths(status="running", model_name=model_name, pattern=pattern)


def get_running_cross_validation_names(model_name: str | None = None, pattern: str | Pattern = "") -> list[str]:
    return [cv_path.name for cv_path in get_running_cross_validation_paths(model_name=model_name, pattern=pattern)]


def get_finished_cross_validation_paths(model_name: str | None = None, pattern: str | Pattern = "") -> list[Path]:
    return _get_status_cross_validation_paths(status="finished", model_name=model_name, pattern=pattern)


def get_finished_cross_validation_names(model_name: str | None = None, pattern: str | Pattern = "") -> list[str]:
    return [cv_path.name for cv_path in get_finished_cross_validation_paths(model_name=model_name, pattern=pattern)]


def _get_status_cross_validation_paths(
    status: Literal["running", "finished"], model_name: str | None = None, pattern: str | Pattern = ""
) -> list[Path]:
    from model import experiment

    func = experiment.is_running if status == "running" else experiment.is_finished
    defined_exps = get_defined_experiment_names(model_name=model_name, pattern=pattern)
    cv_paths = []

    for cv_path in get_cross_validation_paths(model_name=model_name, pattern=pattern):
        cv_dirs = path.get_dirs(cv_path)
        for cv_dir in cv_dirs:
            status_ = []
            if cv_dir.name.split(".")[0] in defined_exps:
                status_.append(func(cv_dir))

        if status == "running" and any(status_):
            cv_paths.append(cv_path)
        elif status == "finished" and all(status_):
            cv_paths.append(cv_path)

    return cv_paths


# early stop
def __activate_early_stop(exp_name: str) -> None:
    try:
        if early_stop_is_active(exp_name):
            print_error(
                f"Failed to activate early stop. Early stop is already active for `{exp_name} ({project.get_model()})`."
            )
            sys.exit(1)

        file.touch(_IS_ACTIVE_PATH_.format(exp_name=exp_name), exists_ok=False, lock=True)
        print_info(f"Early stop activated. The training process will be stopped after this epoch.")
    except (ValueError, RuntimeError) as e:
        print_error(f"Failed to activate early stop. {e.args[0]}")


def __deactivate_early_stop(exp_name: str) -> None:
    try:
        deactivate_early_stop(exp_name)
        print_info(f"Early stop deactivated.")
    except (ValueError, RuntimeError) as e:
        print_error(f"Failed to deactivate early stop. {e.args[0]}")


def __early_stop_info() -> None:
    active_exp_names = [exp_name for exp_name in get_running_experiment_names() if early_stop_is_active(exp_name)]

    if not active_exp_names:
        print_info(f"No early stop found for `{project.get_model()}`.")
    else:
        print(f"Early stop active for `{project.get_model()}`:")
        for exp_name in active_exp_names:
            print(f"\t{exp_name}")


def early_stop_is_active(exp_name: str) -> bool:
    _update_early_stop()

    if not exp_name.split(".")[0] in get_defined_experiment_names():
        raise ValueError(f"Experiment `{exp_name}` not found for `{project.get_model()}`.")

    if not exp_name in get_running_experiment_names():
        raise RuntimeError(f"Experiment `{exp_name}` is not running for `{project.get_model()}`.")

    return Path(_IS_ACTIVE_PATH_.format(exp_name=exp_name)).exists()


def deactivate_early_stop(exp_name: str) -> None:
    if not early_stop_is_active(exp_name):
        raise RuntimeError(
            f"Failed to deactivate early stop. Early stop is not active for `{exp_name} ({project.get_model()})`."
        )

    Path(_IS_ACTIVE_PATH_.format(exp_name=exp_name)).unlink()


def _update_early_stop() -> None:
    if _EARLY_STOP_DIR.exists():
        for early_stop_path in path.get_files(_EARLY_STOP_DIR):
            if not early_stop_path.name in get_running_experiment_names():
                early_stop_path.unlink()
