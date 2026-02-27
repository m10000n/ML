from __future__ import annotations

import copy
import inspect
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, cast, get_type_hints

import isodate
import torch
from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from config.file_names import EXP_FILE_NAME, OVERVIEW_FILE_NAME, TRAINED_MODEL_FILE_NAME
from data import data_loader
from data.data_loader import DataLoaderConfig, DataLoadersConfig
from helper import file, model_, path, statistic, system, time
from helper.class_ import (
    Datasets_,
    DatasetsConfig_,
    Model_,
    ModelConfig_,
)
from helper.exception import DataNotFoundError
from helper.helper_ import round_to_str
from helper.model_ import Model_
from helper.validator import validator
from model.clip_grad import ClipGrad_, ClipGradConfig_
from model.criterion import Criterion_, CriterionConfig_
from model.optimizer import Optimizer_, OptimizerConfig_
from model.scheduler import Scheduler_, SchedulerConfig_
from model.stop_criterion import StopCriterion_, StopCriterionConfig_
from model.warmup import Warmup, WarmupConfig

EXPERIMENT_DATASETS = ["train", "val", "test"]

_EXP_STATUS_LOOKUP = {
    "done": "",
    "running": "~R",
    "early_stop": "~ES",
    "created": "~C",
    "aborted": "~A",
}
_EXP_STATUS_LOOKUP_REVERSE = {v.lstrip("~"): k for k, v in _EXP_STATUS_LOOKUP.items()}
_EXP_STATUS_FINISHED = ["done", "early_stop"]
_EXP_STATUS_LITERAL = Literal["done", "running", "early_stop", "created", "aborted"]


def get_exp_dir(exp_dir: Path, with_status: bool = True) -> Path:
    name = get_exp_name(exp_dir)
    exps_dir = exp_dir.parent

    try:
        status_dir = next(path_ for path_ in path.get_dirs(exps_dir) if path_.name.startswith(name))
    except StopIteration:
        raise FileNotFoundError(f"No experiment with name `{name}` found in `{path.make_relative(exps_dir)}`.")

    return status_dir if with_status else exp_dir.parent / name


def get_exp_name(exp_dir: Path) -> str:
    _sanity_check_exp_name(exp_dir.name)
    return exp_dir.name.split("~")[0]


def get_status_dir(exp_dir: Path) -> _EXP_STATUS_LITERAL:
    _sanity_check_exp_name(exp_dir.name)

    name_split = exp_dir.name.split("~")
    status = "" if len(name_split) == 1 else name_split[-1]

    return cast(_EXP_STATUS_LITERAL, _EXP_STATUS_LOOKUP_REVERSE[status])


def is_running(exp_dir: Path) -> bool:
    return get_status_dir(exp_dir) == "running"


def is_finished(exp_dir: Path) -> bool:
    return get_status_dir(exp_dir) in _EXP_STATUS_FINISHED


def _sanity_check_exp_name(name: str) -> None:
    if name.count("~") > 1:
        raise ValueError(f"The name of an experiment ({name}) must not contain more than one '~'.")


@dataclass
class ExperimentResult:
    strict: bool
    support: list[int]
    accuracy: float | None
    balanced_accuracy: float | None
    macro_f1_score: float | None
    weighted_f1_score: float | None
    precision: list[float] | None
    recall: list[float] | None
    f1_score: list[float] | None

    @staticmethod
    def from_dict(dict_: dict[str, Any]) -> ExperimentResult:
        return ExperimentResult(
            strict=dict_["strict"],
            support=dict_["support"],
            accuracy=dict_["accuracy"],
            balanced_accuracy=dict_["balanced_accuracy"],
            macro_f1_score=dict_["macro_f1_score"],
            weighted_f1_score=dict_["weighted_f1_score"],
            precision=dict_["precision"],
            recall=dict_["recall"],
            f1_score=dict_["f1_score"],
        )


class Experiment:
    # config
    _config: ExperimentConfig  # no default value
    _seed: int | None
    _cross_validation: bool

    # status
    _status: _EXP_STATUS_LITERAL
    _early_stop: Literal["user", "no_learn"] | None
    _aborted: Literal["nan", "keyboard_interrupt", "other"] | None

    # data
    _datasets: Datasets_ | None
    _data_loader_configs: dict[Literal["train", "val", "test"], DataLoaderConfig] | None
    _n_batches: dict[Literal["train", "val", "test"], int | None]
    _data_metadata: list[str]

    # system
    _system: dict[Literal["cpu", "ram", "gpu"], Any]
    _resources: dict[
        Literal["pu", "num_workers", "prefetch_factor", "autocast"], int | list[int] | Literal["cpu"] | bool
    ]

    # time
    _time: dict[Literal["train_start", "train_end", "test_start", "test_end"], datetime | None]
    _track_time_spend: bool
    _time_spend: dict[Literal["loading", "to_pu", "forward", "backward"], timedelta]

    # metric
    _epoch: dict[Literal["total", "saved_model", "duration"], int | list[timedelta] | None]
    _learning_rate: list[float]
    _loss: dict[Literal["warmup", "train", "val", "test"], list[float] | float]
    _val_accuracy: list[float]
    _confusion: dict[Literal["ids", "actual", "predicted"], list[int] | list[str]]
    _result: ExperimentResult | None

    # debug
    _debug: dict[str, Any]

    @validator.constraints("num_workers", "x > 0")
    def __init__(
        self,
        config: ExperimentConfig,
        cross_validation: bool = False,
        pu: list[int] | Literal["cpu"] | None = None,
        num_workers: int | None = None,
        prefetch_factor: int | None = None,
        autocast: bool | None = None,
        track_time: bool = False,
    ) -> None:
        self._set_default_values()
        self._config = config
        self._cross_validation = cross_validation

        if pu is not None:
            self._resources["pu"] = pu
        if num_workers is not None:
            self._resources["num_workers"] = num_workers
        if prefetch_factor is not None:
            self._resources["prefetch_factor"] = prefetch_factor
        if autocast is not None:
            self._resources["autocast"] = autocast

        self._track_time_spend = track_time

        self._create_exp_dir()
        self.write()

    @staticmethod
    def _fill_object(a: Experiment | dict[str, Any], b: dict[str, Any]) -> None:
        for k, v in b.items():
            if isinstance(a, Experiment):
                if not hasattr(a, k):
                    raise ValueError(f"The attribute `{k}` is not found.")

                if isinstance(v, dict):
                    Experiment._fill_object(getattr(a, k), v)
                else:
                    setattr(a, k, v)
            else:
                if not k in a:
                    raise ValueError(f"The key `{k}` is not found.")

                if isinstance(v, dict):
                    Experiment._fill_object(a[k], v)
                else:
                    a[k] = v

    def _set_default_values(self) -> None:
        # config
        self._seed = system.get_seed()
        self._cross_validation = False

        # status
        self._status = "created"
        self._aborted = None
        self._early_stop = None

        # data
        self._datasets = None
        self._data_loader_configs = None
        self._n_batches = {"train": None, "val": None, "test": None}
        self._data_metadata = []

        # system
        self._system = {
            "cpu": system.get_cpu_info(),
            "ram": system.get_ram_info(),
            "gpu": system.get_gpu_info(),
        }
        self._resources = {
            "pu": system.get_pu(),
            "num_workers": system.get_num_workers(),
            "prefetch_factor": system.get_prefetch_factor(),
            "autocast": system.with_autocast(),
        }

        # time
        self._time = {
            "train_start": None,
            "train_end": None,
            "test_start": None,
            "test_end": None,
        }
        self._track_time_spend = False
        self._time_spend = {
            "loading": timedelta(0),
            "to_pu": timedelta(0),
            "forward": timedelta(0),
            "backward": timedelta(0),
        }

        # metric
        self._epoch = {
            "total": -1,
            "saved_model": -1,
            "duration": None,
        }
        self._learning_rate = []
        self._loss = {
            "warmup": [],
            "train": [],
            "val": [],
            "test": -1.0,
        }
        self._val_accuracy = []
        self._confusion = {
            "ids": [],
            "actual": [],
            "predicted": [],
        }
        self._result = None

        # debug
        self._debug = {}

    # load
    @staticmethod
    def load(dir_: Path | str) -> Experiment:
        return Experiment.load_from_dict(dict_=Experiment.load_dict(dir_=dir_), dir_=dir_)

    @staticmethod
    def load_dict(dir_: Path | str) -> dict[str, Any]:
        dir_ = get_exp_dir(Path(dir_), with_status=True)
        return file.read_json(path=dir_ / EXP_FILE_NAME.format(exp_name=get_exp_name(dir_)), unlock=True)

    @staticmethod
    def load_from_dict(dict_: dict[str, Any], dir_: Path | str) -> Experiment:
        dict_ = Experiment._hanlde_old_dict(dict_)
        dict_ = {f"_{k}": v for k, v in dict_.items()}

        exp = object.__new__(Experiment)
        exp._set_default_values()
        exp._config = ExperimentConfig.from_dict(dict_=dict_.pop("_config"), exps_dir=Path(dir_).parent)
        exp._result = (
            None if dict_.get("_result", None) is None else ExperimentResult.from_dict(dict_=dict_.pop("_result"))
        )

        dict_["_time"] = {k: datetime.fromisoformat(v) if v else None for k, v in dict_["_time"].items()}
        dict_["_time_spend"] = {k: isodate.parse_duration(v) for k, v in dict_["_time_spend"].items()}
        dict_["_epoch"]["duration"] = (
            [isodate.parse_duration(duration) for duration in dict_["_epoch"]["duration"]]
            if dict_["_epoch"]["duration"]
            else None
        )

        exp._fill_object(a=exp, b=dict_)

        return exp

    @staticmethod
    def _hanlde_old_dict(dict_: dict[str, Any]) -> dict[str, Any]:
        return dict_

    # reload
    def reload(self) -> None:
        exp_path = get_exp_dir(self.get_dir()) / EXP_FILE_NAME.format(exp_name=self.get_name())
        dict_ = file.read_json(path=exp_path, unlock=True)
        dict_ = {f"_{k}": v for k, v in dict_.items()}

        self._set_default_values()

        self._config = ExperimentConfig.from_dict(dict_=dict_.pop("_config"), exps_dir=self._config.exps_dir)
        self._result = (
            None if dict_.get("_result", None) is None else ExperimentResult.from_dict(dict_=dict_.pop("_result"))
        )

        dict_["_time"] = {k: datetime.fromisoformat(v) if v else None for k, v in dict_["_time"].items()}
        dict_["_time_spend"] = {k: isodate.parse_duration(v) for k, v in dict_["_time_spend"].items()}
        dict_["_epoch"]["duration"] = (
            [isodate.parse_duration(duration) for duration in dict_["_epoch"]["duration"]]
            if dict_["_epoch"]["duration"]
            else None
        )

        self._fill_object(a=self, b=dict_)

    # reset
    def reset(self) -> None:
        values_to_keep = {
            "_config": self._config,
            "_seed": self._seed,
            "_cross_validation": self._cross_validation,
            "_status": self._status,
            "_resources": copy.deepcopy(self._resources),
            "_track_time_spend": self._track_time_spend,
        }

        self._set_default_values()
        self._fill_object(a=self, b=values_to_keep)

        self.set_status("created")
        self.write()

    # flags
    def for_cross_validation(self) -> bool:
        return self._cross_validation

    def track_time_spend(self) -> bool:
        return self._track_time_spend

    def has_warmup(self) -> bool:
        return self._config.train.warmup is not None

    # getters
    def get_seed(self) -> int | None:
        if self._seed is None:
            raise DataNotFoundError("The seed was not recorded for this experiment.")
        return self._seed

    def get_config(self) -> ExperimentConfig:
        return copy.deepcopy(self._config)

    def get_n_batches(self, type: Literal["train", "val", "test"]) -> int:
        if self._n_batches[type] is None:
            raise RuntimeError("Call `download_datasets` before calling `get_n_batches`.")

        return cast(int, self._n_batches[type])

    def get_data_metadata(self) -> list[str]:
        return copy.deepcopy(self._data_metadata)

    def get_system(self) -> dict[Literal["cpu", "ram", "gpu"], Any]:
        return copy.deepcopy(self._system)

    def get_resources(self) -> dict[Literal["pu", "num_workers", "prefetch_factor", "autocast"], Any]:
        return copy.deepcopy(self._resources)

    def get_pu(self) -> list[int] | Literal["cpu"]:
        return cast(list[int] | Literal["cpu"], copy.deepcopy(self._resources["pu"]))

    def get_num_workers(self) -> int:
        return cast(int, self._resources["num_workers"])

    def get_prefetch_factor(self) -> int:
        return cast(int, self._resources["prefetch_factor"])

    def with_autocast(self) -> bool:
        return cast(bool, self._resources["autocast"])

    def get_time(self, run: Literal["train", "test"], event: Literal["start", "end"]) -> datetime | None:
        return self._time[f"{run}_{event}"]  # type: ignore [index]

    def get_duration(self, run: Literal["train", "test"]) -> timedelta:
        self._assert_finished()

        return cast(datetime, self._time[f"{run}_end"]) - cast(datetime, self._time[f"{run}_start"])  # type: ignore[index]

    def get_time_(self) -> dict[Literal["train_start", "train_end", "test_start", "test_end"], datetime | None]:
        return self._time

    def get_time_spend(self, task: Literal["loading", "to_pu", "forward", "backward"]) -> timedelta:
        return self._time_spend[task]

    def get_time_spend_(self) -> dict[Literal["loading", "to_pu", "forward", "backward"], timedelta]:
        return self._time_spend

    def get_total_epochs(self) -> int:
        return cast(int, self._epoch["total"])

    def get_model_epoch(self) -> int:
        return cast(int, self._epoch["saved_model"])

    def get_epoch_duration(self) -> list[timedelta]:
        if self._epoch["duration"] is None:
            raise DataNotFoundError("The duration of each epoch was not recorded for this experiment.")
        return cast(list[timedelta], copy.deepcopy(self._epoch["duration"]))

    def get_learning_rate(self) -> list[float]:
        return copy.deepcopy(self._learning_rate)

    def get_loss(self, select: Literal["warmup", "train", "val", "test"]) -> list[float] | float:
        return copy.deepcopy(self._loss[select])

    def get_loss_(self) -> dict[Literal["warmup", "train", "val", "test"], list[float] | float]:
        return copy.deepcopy(self._loss)

    def get_warmup_loss_as_epochs(self) -> tuple[list[float], float]:
        if not self.has_warmup():
            raise ValueError("This experiment include no warmup.")

        warmup_loss = cast(list[float], self.get_loss("warmup"))

        if len(warmup_loss) == 0:
            return [], 0
        else:
            n_batches = self.get_n_batches("train")
            epochs = [warmup_loss[i : i + n_batches] for i in range(0, len(warmup_loss), n_batches)]
            return [sum(epoch) / len(epoch) for epoch in epochs], len(epochs[-1]) / n_batches

    def get_accuracy(self, dataset: Literal["val", "test"] = "test") -> list[float] | float | None:
        if dataset == "val":
            return copy.deepcopy(self._val_accuracy)
        else:
            self._assert_evaluated()
            result = cast(ExperimentResult, self._result)
            return result.accuracy

    def get_accuracy_(self) -> dict[Literal["val", "test"], list[float] | float | None]:
        self._assert_evaluated()
        result = cast(ExperimentResult, self._result)
        return {
            "val": copy.deepcopy(self._val_accuracy),
            "test": result.accuracy,
        }

    def get_confusion(self) -> dict[Literal["ids", "actual", "predicted"], list[int] | list[str]]:
        return copy.deepcopy(self._confusion)

    def get_confusion_ids(self) -> list[str]:
        return cast(list[str], copy.deepcopy(self._confusion["ids"]))

    def get_confusion_actual(self) -> list[int]:
        return cast(list[int], copy.deepcopy(self._confusion["actual"]))

    def get_confusion_predicted(self) -> list[int]:
        return cast(list[int], copy.deepcopy(self._confusion["predicted"]))

    def get_debug(self) -> dict[str, Any]:
        return copy.deepcopy(self._debug)

    # setters
    def set_seed_to_none(self) -> None:
        self._seed = None

    def set_pu(self, pu: list[int] | Literal["cpu"]) -> None:
        self._resources["pu"] = copy.deepcopy(pu)

    @validator.constraints("num_workers", "x > 0")
    def set_num_workers(self, num_workers: int) -> None:
        self._resources["num_workers"] = num_workers

    @validator.constraints("prefetch_factor", "x > 0")
    def set_prefetch_factor(self, prefetch_factor: int) -> None:
        self._resources["prefetch_factor"] = prefetch_factor

    def set_autocast(self, autocast: bool) -> None:
        self._resources["autocast"] = autocast

    def set_time_spend(self, task: Literal["loading", "to_pu", "forward", "backward"], time_spend: timedelta) -> None:
        self._time_spend[task] = time_spend

    @validator.constraints("n_epochs", "x >= 0")
    def set_total_epochs(self, n_epochs: int) -> None:
        self._epoch["total"] = n_epochs

    @validator.constraints("epoch", "x >= 0")
    def set_model_epoch(self, epoch: int) -> None:
        self._epoch["saved_model"] = epoch

    def set_duration_epoch(self, duration_epoch: list[timedelta]) -> None:
        self._epoch["duration"] = duration_epoch

    @validator.constraints("learning_rate", "x > 0")
    def set_learning_rate(self, learning_rate: list[float]) -> None:
        self._learning_rate = copy.deepcopy(learning_rate)

    @validator.constraints("loss", "x >= 0")
    def set_loss(self, loss: list[float] | float, select: Literal["warmup", "train", "val", "test"]) -> None:
        if select in ["warmup", "train", "val"] and isinstance(loss, float):
            raise ValueError(f"`loss` must be a list if you want to update `{select}`.")
        if select == "test" and isinstance(loss, list):
            raise ValueError(f"`loss` must be a float if you want to update `{select}`.")

        self._loss[select] = copy.deepcopy(loss)

    @validator.constraints("accuracy", "x >= 0")
    def set_val_accuracy(self, accuracy: list[float]) -> None:
        self._val_accuracy = copy.deepcopy(accuracy)

    @validator.constraints("actual", "x >= 0")
    def set_confusion_actual(self, actual: list[int]) -> None:
        self._confusion["actual"] = copy.deepcopy(actual)

    @validator.constraints("predicted", "x >= 0")
    def set_confusion_predicted(self, predicted: list[int]) -> None:
        self._confusion["predicted"] = copy.deepcopy(predicted)

    def set_confusion_ids(self, ids: list[str]) -> None:
        self._confusion["ids"] = copy.deepcopy(ids)

    def set_debug(self, debug: dict[str, Any]) -> None:
        self._debug = copy.deepcopy(debug)

    # append
    def append_epoch_duration(self, duration: timedelta) -> None:
        if not self._epoch["duration"]:
            self._epoch["duration"] = []

        cast(list[timedelta], self._epoch["duration"]).append(duration)

    @validator.constraints("learning_rate", "x > 0")
    def append_learning_rate(self, learning_rate: float) -> None:
        self._learning_rate.append(learning_rate)

    @validator.constraints("loss", "x >= 0")
    def append_loss(self, loss: float, select: Literal["warmup", "train", "val"]) -> None:
        cast(list[float], self._loss[select]).append(loss)

    @validator.constraints("accuracy", "x >= 0")
    def append_val_accuracy(self, accuracy: float) -> None:
        self._val_accuracy.append(accuracy)

    @validator.constraints("actual", "x >= 0")
    def append_confusion_actual(self, actual: int) -> None:
        cast(list[int], self._confusion["actual"]).append(actual)

    @validator.constraints("predicted", "x >= 0")
    def append_confusion_predicted(self, predicted: int) -> None:
        cast(list[int], self._confusion["predicted"]).append(predicted)

    def append_confusion_ids(self, id: str) -> None:
        cast(list[str], self._confusion["ids"]).append(id)

    # add
    def add_time_spend(self, task: Literal["loading", "to_pu", "forward", "backward"], time: timedelta) -> None:
        self._time_spend[task] += time

    # paths
    def get_dir(self, with_status: bool = True) -> Path:
        dir_ = self._config.dir_
        return dir_.with_name(dir_.name + _EXP_STATUS_LOOKUP[self.get_status()]) if with_status else dir_

    def get_plot_dir(self) -> Path:
        return self.get_dir() / "plot"

    def _create_exp_dir(self) -> None:
        dir_ = self.get_dir()
        if dir_.exists():
            raise ValueError(f"The experiment directory `{path.make_relative(self._config.dir_)}` already exists.")
        dir_.mkdir(parents=False)

    def _get_exp_path(self) -> Path:
        return self.get_dir() / EXP_FILE_NAME.format(exp_name=self._config.name)

    def _get_overview_path(self) -> Path:
        return self.get_dir() / OVERVIEW_FILE_NAME.format(exp_name=self._config.name)

    def _get_model_path(self) -> Path:
        return self.get_dir() / TRAINED_MODEL_FILE_NAME.format(exp_name=self._config.name)

    # events
    def start(self, mode: Literal["train", "test"]) -> None:
        if mode == "train":
            self._time["train_start"] = datetime.now()
        else:
            self._time["test_start"] = datetime.now()

    def end(self, mode: Literal["train", "test"]) -> None:
        if mode == "train":
            self._time["train_end"] = datetime.now()
        else:
            self._time["test_end"] = datetime.now()

    # status
    def get_status(self) -> _EXP_STATUS_LITERAL:
        return self._status

    def get_early_stop_reason(self) -> Literal["user", "no_learn"] | None:
        return self._early_stop

    def get_aborted_reason(self) -> Literal["nan", "keyboard_interrupt", "other"] | None:
        return self._aborted

    def is_running(self) -> bool:
        return self.get_status() == "running"

    def is_finished(self) -> bool:
        return self.get_status() in ["done", "early_stop"]

    def is_evaluated(self) -> bool:
        return self._result is not None

    def set_status(self, status: _EXP_STATUS_LITERAL) -> None:
        exp_dir = self.get_dir()
        self._status = status
        exp_dir.rename(self.get_dir())

    def set_early_stop_reason(self, reason: Literal["user", "no_learn"]) -> None:
        self._early_stop = reason

    def set_aborted_reason(self, reason: Literal["nan", "keyboard_interrupt", "other"]) -> None:
        self._aborted = reason

    # config wrappers
    def get_name(self) -> str:
        return self._config.name

    def get_iteration(self) -> int:
        return self._config.iteration

    def get_classes(self) -> list[str]:
        return copy.deepcopy(self._config.data.get_classes(pretty=True))

    def get_n_classes(self) -> int:
        return len(self.get_classes())

    def get_batch_size(self) -> int:
        return self._config.data_loaders.batch_size

    def get_batch_accumulation(self) -> int:
        return self._config.train.batch_accumulation

    def download_datasets(self) -> None:
        datasets = self._get_datasets()
        datasets.download()
        self._data_metadata = datasets.get_metadata()
        self._n_batches = {
            "train": math.ceil(len(datasets.get_dataset("train")) / self.get_batch_size()),  # type: ignore[arg-type]
            "val": math.ceil(len(datasets.get_dataset("val")) / self.get_batch_size()),  # type: ignore[arg-type]
            "test": math.ceil(len(datasets.get_dataset("test")) / self.get_batch_size()),  # type: ignore[arg-type]
        }

    def create_loader(
        self, type: Literal["train", "val", "test"], world_size: int, rank: int, num_workers: int, prefetch_factor: int
    ) -> DataLoader:
        if self._data_loader_configs is None:
            configs = self._config.data_loaders.to_configs(list(self._get_datasets().get_datasets()))
            self._data_loader_configs = {"train": configs[0], "val": configs[1], "test": configs[2]}

        return data_loader.get_loader(
            config=self._data_loader_configs[type],
            world_size=world_size,
            rank=rank,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def create_model(self, pu: device) -> Model_:
        return Model_.create(self._config.model).to(pu)

    def load_model(self, pu: device) -> Model_:
        model_path = self._get_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"The model file `{path.make_relative(model_path)}` does not exist.")

        model = self.create_model(pu)
        model.load_state_dict(torch.load(f=model_path, map_location="cpu"), strict=True)
        return model

    def create_criterion(self, pu: device) -> Criterion_:
        if self._config.criterion.class_weighting:
            self._config.criterion.set_class_weights(self._get_datasets().get_class_distribution())

        criterion = Criterion_.create(self._config.criterion)

        if self._config.criterion.class_weighting:
            criterion.class_weights_to(pu)

        return criterion

    def create_optimizer(self, model: Module) -> Optimizer_:
        return Optimizer_.create(self._config.train.optimizer, model)

    def create_scheduler(self, optimizer: Optimizer_) -> Scheduler_:
        return Scheduler_.create(self._config.train.scheduler, optimizer)

    def create_stop_criterion(self) -> StopCriterion_:
        return StopCriterion_.create(self._config.train.stop_criterion)

    def create_clip_grad(self, model: Module) -> ClipGrad_ | None:
        return ClipGrad_.create(self._config.train.clip_grad, model) if self._config.train.clip_grad else None

    def create_warmup(self, optimizer: Optimizer_) -> Warmup | None:
        return (
            None if self._config.train.warmup is None else Warmup(config=self._config.train.warmup, optimizer=optimizer)
        )

    def get_no_learn_limit(self) -> int | None:
        return self._config.train.no_learn_limit

    def _get_datasets(self) -> Datasets_:
        if self._datasets is None:
            self._datasets = Datasets_.create(self._config.data)
        return self._datasets

    # result
    def evaluate(self, strict: bool = True) -> None:
        self._assert_finished()

        ids = self.get_confusion_ids()
        actual = self.get_confusion_actual()
        predicted = self.get_confusion_predicted()
        n_ids = len(ids)
        n_actual = len(actual)
        n_predicted = len(predicted)

        if n_ids == 0:
            raise ValueError("No confusion ids provided.")

        if n_actual == 0:
            raise ValueError("No actual labels provided.")

        if n_predicted == 0:
            raise ValueError("No predicted labels provided.")

        if n_actual != n_predicted != n_ids:
            raise ValueError("The number of confusion ids, actual labels and predicted labels must be the same.")

        n_classes = self.get_n_classes()

        support = statistic.get_support(actual=actual, n_classes=n_classes)

        accuracy = statistic.calculate_accuracy(
            actual=actual,
            predicted=predicted,
            n_classes=n_classes,
        )

        try:
            balanced_accuracy = statistic.calculate_balanced_accuracy(
                actual=actual,
                predicted=predicted,
                n_classes=n_classes,
                strict=strict,
            )
        except ValueError:
            balanced_accuracy = None

        try:
            macro_f1_score = statistic.calculate_macro_f1_score(
                actual=actual, predicted=predicted, n_classes=n_classes, strict=strict
            )
        except ValueError:
            macro_f1_score = None

        try:
            weighted_f1_score = statistic.calculate_weighted_f1_score(
                actual=actual, predicted=predicted, n_classes=n_classes, strict=strict
            )
        except ValueError:
            weighted_f1_score = None

        try:
            precision = statistic.calculate_precision(
                actual=actual, predicted=predicted, n_classes=n_classes, strict=strict
            )
        except ValueError:
            precision = None

        try:
            recall = statistic.calculate_recall(actual=actual, predicted=predicted, n_classes=n_classes, strict=strict)
        except ValueError:
            recall = None

        try:
            f1_score = statistic.calculate_f1_score(
                actual=actual, predicted=predicted, n_classes=n_classes, strict=strict
            )
        except ValueError:
            f1_score = None

        self._result = ExperimentResult(
            strict=strict,
            support=support,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            macro_f1_score=macro_f1_score,
            weighted_f1_score=weighted_f1_score,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

        self.write()

    def get_support(self) -> list[int]:
        self._assert_finished()
        result = cast(ExperimentResult, self._result)
        return copy.deepcopy(result.support)

    def get_balanced_accuracy(self) -> float | None:
        self._assert_finished()
        result = cast(ExperimentResult, self._result)
        return result.balanced_accuracy

    def get_macro_f1_score(self) -> float | None:
        self._assert_finished()
        result = cast(ExperimentResult, self._result)
        return result.macro_f1_score

    def get_weighted_f1_score(self) -> float | None:
        self._assert_finished()
        result = cast(ExperimentResult, self._result)
        return result.weighted_f1_score

    def get_precision(self) -> list[float] | None:
        self._assert_finished()
        result = cast(ExperimentResult, self._result)
        return copy.deepcopy(result.precision)

    def get_recall(self) -> list[float] | None:
        self._assert_finished()
        result = cast(ExperimentResult, self._result)
        return copy.deepcopy(result.recall)

    def get_f1_score(self) -> list[float] | None:
        self._assert_finished()
        result = cast(ExperimentResult, self._result)
        return copy.deepcopy(result.f1_score)

    def get_metric(
        self,
        metric: Literal[
            "accuracy", "balanced_accuracy", "macro_f1_score", "weighted_f1_score", "precision", "recall", "f1_score"
        ],
    ) -> float | list[float] | None:
        self._assert_finished()
        return getattr(self._result, metric)

    def get_result_str(self, ljust: int = 0) -> str:
        self._assert_evaluated()
        result = cast(ExperimentResult, self._result)
        na = "N/A"

        accuracy_str = f"{round_to_str(x=result.accuracy, digits=3)}" if result.accuracy else na
        accuracy_str = accuracy_str.ljust(5)

        balanced_accuracy_str = (
            f"{round_to_str(x=result.balanced_accuracy, digits=3)}" if result.balanced_accuracy else na
        )
        balanced_accuracy_str = balanced_accuracy_str.ljust(5)

        macro_f1_score_str = f"{round_to_str(x=result.macro_f1_score, digits=3)}" if result.macro_f1_score else na
        macro_f1_score_str = macro_f1_score_str.ljust(5)

        weighted_f1_score_str = (
            f"{round_to_str(x=result.weighted_f1_score, digits=3)}" if result.weighted_f1_score else na
        )
        weighted_f1_score_str = weighted_f1_score_str.ljust(5)

        loss_str = f"{round_to_str(x=cast(float, self.get_loss('test')), digits=4)}".ljust(6)

        if self.get_status() == "early_stop":
            status_str = f", early stop ({self.get_early_stop_reason()})"
        else:
            status_str = ""

        return (
            f"{self.get_name().ljust(ljust)} - "
            f"strict: {"T" if result.strict else "F"}, "
            f"acc: {accuracy_str}, "
            f"b acc: {balanced_accuracy_str}, "
            f"m f1: {macro_f1_score_str}, "
            f"w f1: {weighted_f1_score_str}, "
            f"loss: {loss_str}, "
            f"epochs: {self.get_total_epochs()}"
            f"{status_str}"
        )

    # write
    def write(self) -> None:
        file.write_json(path=self._get_exp_path(), data=self.as_dict(), overwrite=True, lock=True)

    # overview
    def create_overview(self) -> None:
        self._assert_evaluated()

        na = "N/A"
        overview_path = self._get_overview_path()

        train_duration = (
            self._time["train_end"] - self._time["train_start"]
            if self._time["train_start"] and self._time["train_end"]
            else timedelta(0)
        )
        test_duration = (
            self._time["test_end"] - self._time["test_start"]
            if self._time["test_start"] and self._time["test_end"]
            else timedelta(0)
        )
        total_duration = test_duration + train_duration

        total_s = total_duration.total_seconds()

        if self._early_stop:
            status_reason = str(self._early_stop)
        elif self._aborted:
            status_reason = str(self._aborted)
        else:
            status_reason = None

        overview = dedent(
            f"""\
            experiment name: {self._config.name}
            experiment description: {self._config.description}
            status: {self.get_status()}{f" ({status_reason})" if status_reason else ""}
            model description: {self._config.model.description}
            dataset description: {self._config.data.description}
            data description: {self._config.data.data_config.name}; {self._config.data.data_config.description}

            training duration: {time.h_min_sec_str(train_duration)}
            test duration: {time.h_min_sec_str(test_duration)}
            total duration: {time.h_min_sec_str(total_duration)}
            """
        )

        if self._time_spend["loading"] > timedelta(0) and total_s > 0:
            overview += f"spend loading: {round_to_str(x=self._time_spend["loading"].total_seconds() * 100 / total_s, digits=2)}%\n"

        if self._time_spend["to_pu"] > timedelta(0) and total_s > 0:
            overview += f"spend moving to pu: {round_to_str(x=self._time_spend["to_pu"].total_seconds() * 100 / total_s, digits=2)}%\n"

        if self._time_spend["forward"] > timedelta(0) and total_s > 0:
            overview += f"spend forward: {round_to_str(x=self._time_spend["forward"].total_seconds() * 100 / total_s, digits=2)}%\n"

        if self._time_spend["backward"] > timedelta(0) and total_s > 0:
            overview += f"spend backward: {round_to_str(x=self._time_spend["backward"].total_seconds() * 100 / total_s, digits=2)}%\n"

        if self._loss["train"]:
            train_loss = cast(list[float], self._loss["train"])
            epoch = train_loss.index(min(train_loss))
            min_train_loss_str = f"{round_to_str(x=train_loss[epoch], digits=4)} (epoch: {epoch})"
        else:
            min_train_loss_str = na

        if self._loss["val"]:
            val_loss = cast(list[float], self._loss["val"])
            epoch = val_loss.index(min(val_loss))
            min_val_loss_str = f"{round_to_str(x=val_loss[epoch], digits=4)} (epoch: {epoch})"
        else:
            min_val_loss_str = na

        if self._val_accuracy:
            epoch = self._val_accuracy.index(max(self._val_accuracy))
            max_val_accuracy_str = f"{round_to_str(x=self._val_accuracy[epoch], digits=4)} (epoch: {epoch})"
        else:
            max_val_accuracy_str = na

        result = cast(ExperimentResult, self._result)

        accuracy_str = f"{round_to_str(x=result.accuracy, digits=4)}" if result.accuracy else na
        balanced_accuracy_str = (
            f"{round_to_str(x=result.balanced_accuracy, digits=4)}" if result.balanced_accuracy else na
        )
        macro_f1_str = f"{round_to_str(x=result.macro_f1_score, digits=4)}" if result.macro_f1_score else na
        weighted_f1_str = f"{round_to_str(x=result.weighted_f1_score, digits=4)}" if result.weighted_f1_score else na

        overview += dedent(
            f"""
                number of epochs: {self._epoch["total"] if cast(int, self._epoch["total"]) > 0 else na}
                model saved at epoch: {self._epoch["saved_model"] if cast(int, self._epoch["saved_model"]) >= 0 else na}
                lowest training loss: {min_train_loss_str}
                lowest validation loss: {min_val_loss_str}
                highest validation accuracy: {max_val_accuracy_str}
                strict evaluation: {str(result.strict)}
                accuracy: {accuracy_str}
                balanced accuracy: {balanced_accuracy_str}
                macro f1 score: {macro_f1_str}
                weighted f1 score: {weighted_f1_str}

                dataset split: {self._config.data.split}
                dataset metadata:
                """
        )
        overview += "\t" + "\n\t".join(self._data_metadata)

        file.write(path=overview_path, text=overview, overwrite=True, lock=True)

    # as dictionary
    def as_dict(self) -> dict[str, Any]:
        dict_ = {
            "config": self._config.as_dict(),
            "cross_validation": self.for_cross_validation(),
            "seed": self._seed,
            "status": self.get_status(),
            "early_stop": self._early_stop,
            "aborted": self._aborted,
            "n_batches": self._n_batches,
            "data_metadata": self._data_metadata,
            "system": self._system,
            "resources": self._resources,
            "time": {key: value.isoformat() if value else None for key, value in self._time.items()},
            "track_time_spend": self._track_time_spend,
            "time_spend": {key: isodate.duration_isoformat(value) for key, value in self._time_spend.items()},
            "epoch": {
                "total": self._epoch["total"],
                "saved_model": self._epoch["saved_model"],
                "duration": (
                    [
                        isodate.duration_isoformat(duration)
                        for duration in cast(list[timedelta], self._epoch["duration"])
                    ]
                    if self._epoch["duration"]
                    else None
                ),
            },
            "learning_rate": self._learning_rate,
            "loss": self._loss,
            "val_accuracy": self._val_accuracy,
            "confusion": self._confusion,
        }

        if self._result is not None:
            dict_["result"] = asdict(self._result)

        dict_["debug"] = self._debug

        return dict_

    def _assert_finished(self) -> None:
        if not self.is_finished():
            raise RuntimeError("The experiment is not finished.")

    def _assert_evaluated(self) -> None:
        if not self.is_evaluated():
            raise RuntimeError("The experiment has not been evaluated. Call `evaluate` first.")


@dataclass
class ExperimentConfig:
    _name: str = field(init=False)
    iteration: int = field(init=False)
    exps_dir: Path = field(init=False)
    description: str
    model: ModelConfig_
    criterion: CriterionConfig_
    train: TrainConfig
    data: DatasetsConfig_
    data_loaders: DataLoadersConfig

    def __post_init__(self) -> None:
        stack = inspect.stack()
        caller_name = stack[2].function

        if "~" in caller_name:
            raise ValueError(f"The name of an experiment must not contain `~`.")

        if not self.data.is_for_exp():
            raise ValueError("The datasets configuration must contain train, val, and test sets.")

        self._name = caller_name

        runs = model_.get_experiment_names(pattern=rf"^{self._name}\.\d+.*$")
        self.iteration = max([int(exp.split(".")[-1]) for exp in runs]) + 1 if runs else 0

        self.exps_dir = path.experiment(absolute=True)

    @property
    def name(self) -> str:
        return f"{self._name}.{self.iteration}"

    @property
    def name_no_iteration(self) -> str:
        return self._name

    @property
    def dir_(self) -> Path:
        return self.exps_dir / self.name

    @staticmethod
    def from_dict(dict_: dict[str, Any], exps_dir: Path) -> ExperimentConfig:
        config = object.__new__(ExperimentConfig)
        name, iteration = dict_["name"].rsplit(".", 1)

        config._name = name
        config.iteration = int(iteration)
        config.exps_dir = exps_dir
        config.description = dict_["description"]
        config.model = ModelConfig_.from_dict(dict_["model"])
        config.criterion = CriterionConfig_.from_dict(dict_["criterion"])
        config.train = TrainConfig.from_dict(dict_["train"])
        config.data = DatasetsConfig_.from_dict(dict_["data"])
        config.data_loaders = DataLoadersConfig.from_dict(dict_["data_loaders"])

        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model.as_dict(),
            "criterion": self.criterion.as_dict(),
            "train": self.train.as_dict(),
            "data": self.data.as_dict(),
            "data_loaders": self.data_loaders.as_dict(),
        }

    def get_ids(self) -> list[str]:
        return self.data.get_ids()

    def get_info(self, details: bool) -> list[str]:
        info = [f"{self.name_no_iteration}"]

        if details:
            info.extend(
                [
                    f"\n\texp - {self.description}\n"
                    f"\tmodel - {self.model.description}\n"
                    f"\tdatasets - {self.data.description}\n"
                    f"\tdata - {self.data.data_config.name}"
                    f"{', ' + self.data.data_config.description if self.data.data_config.description else ''}"
                ]
            )

        return info


@dataclass
class TrainConfig:
    optimizer: OptimizerConfig_
    scheduler: SchedulerConfig_
    stop_criterion: StopCriterionConfig_
    batch_accumulation: int = 1
    clip_grad: ClipGradConfig_ | None = None
    warmup: WarmupConfig | None = None
    no_learn_limit: int | None = None

    def __post_init__(self) -> None:
        if self.batch_accumulation is not None and self.batch_accumulation < 1:
            raise ValueError("`batch_accumulation` must be positive.")

        if self.no_learn_limit is not None and self.no_learn_limit < 1:
            raise ValueError("`no_learn_limit` must be positive.")

    @staticmethod
    def from_dict(dict_: dict[str, Any]) -> TrainConfig:
        return TrainConfig(
            optimizer=OptimizerConfig_.from_dict(dict_["optimizer"]),
            scheduler=SchedulerConfig_.from_dict(dict_["scheduler"]),
            stop_criterion=StopCriterionConfig_.from_dict(dict_["stop_criterion"]),
            batch_accumulation=dict_["batch_accumulation"],
            clip_grad=ClipGradConfig_.from_dict(dict_["clip_grad"]) if dict_.get("clip_grad") is not None else None,
            warmup=WarmupConfig.from_dict(dict_["warmup"]) if dict_["warmup"] is not None else None,
            no_learn_limit=dict_["no_learn_limit"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.as_dict(),
            "scheduler": self.scheduler.as_dict(),
            "stop_criterion": self.stop_criterion.as_dict(),
            "batch_accumulation": self.batch_accumulation,
            "clip_grad": self.clip_grad.as_dict() if self.clip_grad else None,
            "warmup": self.warmup.as_dict() if self.warmup else None,
            "no_learn_limit": self.no_learn_limit,
        }


_TYPE_HINTS = get_type_hints(Experiment)
