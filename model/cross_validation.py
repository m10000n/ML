from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Literal, cast

from config import file_names
from helper import file, path, statistic, system
from model.experiment import Experiment, ExperimentConfig


@dataclass
class CI:
    low: float
    high: float

    @staticmethod
    def from_dict(dict_: dict) -> CI:
        return CI(dict_["low"], dict_["high"])


@dataclass
class FoldResult:
    per_run: list[float] | list[list[float]]
    mean: float | list[float]
    std: float | list[float]
    ci: CI | list[CI]

    def __init__(self, per_run: list[float] | list[list[float]]):
        self.per_run = per_run

        if isinstance(per_run[0], list):
            per_class = [list(class_) for class_ in zip(*per_run)]
            self.mean = [statistic.calculate_mean(class_) for class_ in per_class]
            self.std = [statistic.calculate_std(class_) for class_ in per_class]
            self.ci = [CI(*statistic.calculate_ci(class_)) for class_ in per_class]
        else:
            per_run_ = cast(list[float], per_run)
            self.mean = statistic.calculate_mean(per_run_)
            self.std = statistic.calculate_std(per_run_)
            self.ci = CI(*statistic.calculate_ci(per_run_))

    @staticmethod
    def from_dict(dict_: dict) -> FoldResult:
        result = object.__new__(FoldResult)
        result.per_run = dict_["per_run"]
        result.mean = dict_["mean"]
        result.std = dict_["std"]
        result.ci = (
            [CI.from_dict(ci) for ci in dict_["ci"]] if isinstance(dict_["ci"], list) else CI.from_dict(dict_["ci"])
        )

        return result


@dataclass
class BootstrapResult:
    point: float | list[float]
    ci: CI | list[CI]

    def __init__(
        self,
        metric: statistic.METRICS_LITERAL,
        ids: list[str],
        actual: list[int],
        predicted: list[int],
        n_classes: int,
        n_resamples: int,
        confidence_level: float,
        seed: int,
    ):
        point, low, high = statistic.bca_ci(
            ids=ids,
            actual=actual,
            predicted=predicted,
            n_classes=n_classes,
            metric=metric,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        self.point = point
        if isinstance(low, list) and isinstance(high, list):
            self.ci = [CI(cast(float, low_), cast(float, high_)) for low_, high_ in zip(low, high)]
        else:
            self.ci = CI(cast(float, low), cast(float, high))

    @staticmethod
    def from_dict(dict_: dict) -> BootstrapResult:
        result = object.__new__(BootstrapResult)
        result.point = dict_["point"]
        result.ci = (
            [CI.from_dict(ci) for ci in dict_["ci"]] if isinstance(dict_["ci"], list) else CI.from_dict(dict_["ci"])
        )
        return result


@dataclass
class CVMetric:
    fold_result: FoldResult
    bootstrap_result: BootstrapResult

    @staticmethod
    def from_dict(dict_: dict) -> CVMetric:
        return CVMetric(
            fold_result=FoldResult.from_dict(dict_["fold_result"]),
            bootstrap_result=BootstrapResult.from_dict(dict_["bootstrap_result"]),
        )


@dataclass
class CrossValidationResult:
    confidence_level: float
    n_resamples: int
    accuracy: CVMetric
    balanced_accuracy: CVMetric
    macro_f1_score: CVMetric
    weighted_f1_score: CVMetric
    precision: CVMetric
    recall: CVMetric
    f1_score: CVMetric
    support: list[int]
    confusion: dict[Literal["ids", "actual", "predicted"], list[str] | list[int]]

    def __init__(
        self,
        runs: list[Experiment],
        n_classes: int,
        confidence_level: float | None = None,
        n_resamples: int | None = None,
        seed: int | None = None,
    ):
        if len(runs) < 2:
            raise ValueError("There must be at least 2 runs.")

        not_evaluated = [cv_run for cv_run in runs if not cv_run.is_evaluated()]
        if not_evaluated:
            not_evaluated_str = ", ".join([cv_run.get_name() for cv_run in not_evaluated])
            raise ValueError(
                f"The following experiment{"s" if len(not_evaluated) > 1 else ""} are not evaluated: {not_evaluated_str}."
            )

        confidence_level_ = confidence_level if confidence_level is not None else statistic.DEFAULT_CONFIDENCE_LEVEL
        n_resamples_ = n_resamples if n_resamples is not None else statistic.DEFAULT_N_RESAMPLES
        seed_ = seed if seed is not None else system.get_seed()

        ids = []
        actual = []
        predicted = []
        support = []

        metrics_per_run: dict[statistic.METRICS_LITERAL, list[float | list[float]]] = {
            cast(statistic.METRICS_LITERAL, metric): [] for metric in statistic.METRICS_LIST
        }

        for run in runs:
            ids_ = run.get_confusion_ids()
            actual_ = run.get_confusion_actual()
            predicted_ = run.get_confusion_predicted()

            ids.extend(ids_)
            predicted.extend(predicted_)
            actual.extend(actual_)
            support.append(run.get_support())

            for name, values in metrics_per_run.items():
                values.append(
                    statistic.calculate_metric(
                        metric=name, actual=actual_, predicted=predicted_, n_classes=n_classes, strict=False
                    )
                )

        self.confidence_level = confidence_level_
        self.n_resamples = n_resamples_

        for name, values in metrics_per_run.items():
            fold_result = FoldResult(cast(list[float] | list[list[float]], values))
            bootstrap_result = BootstrapResult(
                metric=name,
                ids=ids,
                actual=actual,
                predicted=predicted,
                n_classes=n_classes,
                n_resamples=n_resamples_,
                confidence_level=confidence_level_,
                seed=seed_,
            )

            setattr(self, name, CVMetric(fold_result=fold_result, bootstrap_result=bootstrap_result))

        self.support = [sum(class_) for class_ in zip(*support)]
        self.confusion = {"ids": ids, "actual": actual, "predicted": predicted}

    @staticmethod
    def from_dict(dict_: dict) -> CrossValidationResult:
        result = object.__new__(CrossValidationResult)
        result.confidence_level = dict_["confidence_level"]
        result.n_resamples = dict_["n_resamples"]
        result.accuracy = CVMetric.from_dict(dict_["accuracy"])
        result.balanced_accuracy = CVMetric.from_dict(dict_["balanced_accuracy"])
        result.macro_f1_score = CVMetric.from_dict(dict_["macro_f1_score"])
        result.weighted_f1_score = CVMetric.from_dict(dict_["weighted_f1_score"])
        result.precision = CVMetric.from_dict(dict_["precision"])
        result.recall = CVMetric.from_dict(dict_["recall"])
        result.f1_score = CVMetric.from_dict(dict_["f1_score"])
        result.support = dict_["support"]
        result.confusion = dict_["confusion"]
        return result


class CrossValidation:
    _name: str
    iteration: int
    cvs_dir: Path
    class_names: list[str]
    seed: int
    runs: list[Experiment]
    results: CrossValidationResult | None = None

    def __init__(self, exp_config: ExperimentConfig, track_time: bool, seed: int | None = None):
        split = exp_config.data.split
        test_fraction = split[2]
        n_folds = round(1 / test_fraction)

        if not math.isclose(n_folds, 1 / test_fraction):
            raise ValueError(f"The test set fraction ({test_fraction}) must be 1/k for some integer k.")

        ids = exp_config.get_ids()
        n_ids = len(ids)

        if n_ids < n_folds:
            raise ValueError(f"The number of ids ({n_ids}) must be >= the number of folds ({n_folds}).")

        seed_ = seed if seed is not None else system.get_seed()

        self._name = exp_config.name_no_iteration
        self.class_names = exp_config.data.get_classes(pretty=True)

        self.cvs_dir = path.cross_validation()
        self.cvs_dir.mkdir(exist_ok=True)

        iterations = [
            int(cv_run.name.split(".")[1])
            for cv_run in path.get_dirs(path.cross_validation())
            if cv_run.name.split(".")[0] == self.name_no_iteration
        ]
        self.iteration = max(iterations) + 1 if iterations else 0

        self.dir_.mkdir()
        self.seed = seed_

        rng = system.get_rng(seed_)
        rng.shuffle(ids)
        folds = [ids[i::n_folds] for i in range(n_folds)]

        self.runs = []
        for i in range(n_folds):
            test = folds[i]
            train_val = list(chain.from_iterable(folds[:i] + folds[i + 1 :]))
            rng.shuffle(train_val)
            val = train_val[: int(len(train_val) * split[1] / (split[0] + split[1]))]
            train = train_val[len(val) :]
            dataset_ids = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

            config_ = copy.deepcopy(exp_config)
            config_.data.dataset_ids = dataset_ids
            config_.exps_dir = self.dir_
            config_.iteration = i
            self.runs.append(Experiment(config=config_, cross_validation=True, track_time=track_time))

        self.write()

    @staticmethod
    def load(cv_dir: Path | str) -> CrossValidation:
        cv_dir_ = Path(cv_dir)

        if not cv_dir_.is_absolute():
            cv_dir_ = path.make_absolute(cv_dir_)

        cv_json = file.read_json(path=cv_dir_ / file_names.CV_FILE_NAME.format(exp_name=cv_dir_.name), unlock=True)
        cv = object.__new__(CrossValidation)

        name_split = cv_json["name"].split(".")

        cv._name = name_split[0]
        cv.iteration = int(name_split[1])
        cv.cvs_dir = cv_dir_.parent
        cv.class_names = cv_json["class_names"]
        cv.seed = cv_json["seed"]
        cv.runs = [Experiment.load(cv_dir_ / f"{cv.name_no_iteration}.{i}") for i in range(cv_json["n_folds"])]
        cv.results = CrossValidationResult.from_dict(cv_json["results"]) if "results" in cv_json else None
        return cv

    @property
    def name(self) -> str:
        return f"{self._name}.{self.iteration}"

    @property
    def name_no_iteration(self) -> str:
        return self._name

    @property
    def dir_(self) -> Path:
        return self.cvs_dir / self.name

    def is_evaluated(self) -> bool:
        return self.results is not None

    def all_runs_finished(self, reload: bool = True) -> bool:
        if reload:
            self.reload_runs()

        return all(run.is_finished() for run in self.runs)

    def get_n_runs(self) -> int:
        return len(self.runs)

    def get_n_classes(self) -> int:
        return len(self.class_names)

    def get_plot_dir(self) -> Path:
        return self.dir_ / "plot"

    def reload_runs(self) -> None:
        for run in self.runs:
            run.reload()

    def evaluate(
        self, confidence_level: float | None = None, n_resamples: int | None = None, reload: bool = True
    ) -> None:
        if reload:
            self.reload_runs()

        self.results = CrossValidationResult(
            runs=self.runs,
            n_classes=self.get_n_classes(),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=self.seed,
        )
        self.write()

    def as_dict(self) -> dict:
        result = {
            "name": self.name,
            "class_names": self.class_names,
            "seed": self.seed,
            "n_folds": len(self.runs),
        }
        if self.results is not None:
            result["results"] = asdict(self.results)
        return result

    def write(self) -> None:
        file.write_json(
            path=self.dir_ / file_names.CV_FILE_NAME.format(exp_name=self.name),
            data=self.as_dict(),
            overwrite=True,
            lock=True,
        )
