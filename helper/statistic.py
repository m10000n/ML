from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from helper import system
from helper.validator import validator

##### config start #####
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_N_RESAMPLES = 5000
##### config end #####

METRICS_LITERAL = Literal[
    "accuracy", "balanced_accuracy", "macro_f1_score", "weighted_f1_score", "precision", "recall", "f1_score"
]
METRICS_LIST = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1_score",
    "weighted_f1_score",
    "precision",
    "recall",
    "f1_score",
]


def calculate_metric(
    metric: METRICS_LITERAL, actual: list[int], predicted: list[int], n_classes: int, strict: bool = True
) -> float | list[float]:
    cm = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    value = METRIC_MAP[metric](cm=cm, strict=strict)
    if is_class_metric(metric):
        return value.tolist()
    else:
        return float(value)


# confusion matrix
def create_confusion_matrix(actual: list[int], predicted: list[int], n_classes: int) -> NDArray[np.int64]:
    _sanity_check(actual=actual, predicted=predicted, n_classes=n_classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.array(actual, dtype=np.int64), np.array(predicted, dtype=np.int64)), 1)
    return cm


def create_normalized_confusion_matrix(actual: list[int], predicted: list[int], n_classes: int) -> NDArray[np.float64]:
    cm = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes).astype(np.float64)
    support = cm.sum(axis=-1, keepdims=True)

    if np.any(support == 0):
        raise ValueError("Normalized confusion matrix is undefined, at least one class has no support.")

    return cm / support


# support
def get_support(actual: list[int], n_classes: int) -> list[int]:
    return np.bincount(actual, minlength=n_classes).tolist()


# metrics
def calculate_accuracy(actual: list[int], predicted: list[int], n_classes: int) -> float:
    cm = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    return float(_accuracy_from_confusion(cm=cm))


def calculate_balanced_accuracy(actual: list[int], predicted: list[int], n_classes: int, strict: bool = True) -> float:
    cm = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    return float(_balanced_accuracy_from_confusion(cm=cm, strict=strict))


def calculate_macro_f1_score(actual: list[int], predicted: list[int], n_classes: int, strict: bool = True) -> float:
    cm = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    return float(_macro_f1_score_from_confusion(cm=cm, strict=strict))


def calculate_weighted_f1_score(actual: list[int], predicted: list[int], n_classes: int, strict: bool = True) -> float:
    cm = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    return float(_weighted_f1_score_from_confusion(cm=cm, strict=strict))


# per-class metrics
def calculate_precision(actual: list[int], predicted: list[int], n_classes: int, strict: bool = True) -> list[float]:
    confusion_matrix = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    return _precision_from_confusion(cm=confusion_matrix, strict=strict).tolist()


def calculate_recall(actual: list[int], predicted: list[int], n_classes: int, strict: bool = True) -> list[float]:
    confusion_matrix = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    return _recall_from_confusion(cm=confusion_matrix, strict=strict).tolist()


def calculate_f1_score(actual: list[int], predicted: list[int], n_classes: int, strict: bool = True) -> list[float]:
    cm = create_confusion_matrix(actual=actual, predicted=predicted, n_classes=n_classes)
    return _f1_score_from_confusion(cm=cm, strict=strict).tolist()


# statistics
def calculate_mean(values: list[float]) -> float:
    return float(np.mean(values))


def calculate_std(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("There must be at least 2 values.")
    return float(np.std(values, ddof=1))


def calculate_ci(values: list[float], confidence_level: float = DEFAULT_CONFIDENCE_LEVEL) -> tuple[float, float]:
    n = len(values)

    if n < 2:
        raise ValueError("There must be at least 2 values.")

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    if np.isclose(std, 0.0):
        low = mean
        high = mean
    else:
        sem = std / np.sqrt(n)
        low, high = stats.t(df=n - 1, loc=mean, scale=sem).interval(confidence_level)
    return float(low), float(high)


@validator.constraints("tail_fraction", "x > 0 and x < 1")
def tail_instability(values: list[float], tail_fraction: float = 0.5, relative: bool = False) -> float:
    n_values = len(values)

    if n_values < 3:
        raise ValueError("The length of `values` must be at least 3.")

    start = int((1 - tail_fraction) * n_values)
    start = min(max(start, 1), n_values - 2)
    tail = np.asarray(values[start:], dtype=np.float32)
    mean_abs_delta = np.mean(np.abs(np.diff(tail)))

    if relative:
        return float(mean_abs_delta / (np.abs(np.mean(tail)) + 1e-8))
    else:
        return float(mean_abs_delta)


# correlation
def error_pattern_corr(
    cm1: NDArray[np.int64],
    cm2: NDArray[np.int64],
    method: Literal["pearson", "spearman"] = "pearson",
    exclude_diagonal: bool = False,
    normalize: Literal["row", "column", "none"] = "none",
) -> tuple[float, float]:
    if cm1.ndim != 2:
        raise ValueError("`cm1` must have 2 dimensions.")

    if cm2.ndim != 2:
        raise ValueError("`cm2` must have 2 dimensions.")

    if cm1.shape[0] != cm1.shape[1]:
        raise ValueError("`cm1` must be a square matrix.")

    if cm2.shape[0] != cm2.shape[1]:
        raise ValueError("`cm2` must be a square matrix.")

    if cm1.shape != cm2.shape:
        raise ValueError("`cm1` and `cm2` must have the same shape.")

    cm1_ = np.array(cm1, dtype=np.float64, copy=True)
    cm2_ = np.array(cm2, dtype=np.float64, copy=True)

    shape = cm1_.shape

    if exclude_diagonal:
        np.fill_diagonal(cm1_, 0.0)
        np.fill_diagonal(cm2_, 0.0)
        mask = ~np.eye(shape[0], dtype=bool)
    else:
        mask = np.ones(shape, dtype=bool)

    if normalize != "none":
        if normalize == "row":
            cm1_sum = cm1_.sum(axis=1, keepdims=True)
            cm2_sum = cm2_.sum(axis=1, keepdims=True)
        else:
            cm1_sum = cm1_.sum(axis=0, keepdims=True)
            cm2_sum = cm2_.sum(axis=0, keepdims=True)

        cm1_ = np.divide(cm1_, cm1_sum, where=cm1_sum > 0)
        cm2_ = np.divide(cm2_, cm2_sum, where=cm2_sum > 0)

        if normalize == "row":
            cm1_[cm1_sum.flatten() == 0, :] = np.nan
            cm2_[cm2_sum.flatten() == 0, :] = np.nan
        else:
            cm1_[:, cm1_sum.flatten() == 0] = np.nan
            cm2_[:, cm2_sum.flatten() == 0] = np.nan

    valid = mask & np.isfinite(cm1_) & np.isfinite(cm2_)
    cm1_valid = cm1_[valid].ravel()
    cm2_valid = cm2_[valid].ravel()

    if cm1_valid.size < 3:
        raise ValueError("Not enough valid entries to compute correlation.")

    if method == "pearson":
        return stats.pearsonr(cm1_valid, cm2_valid)
    else:
        return stats.spearmanr(cm1_valid, cm2_valid)


# bootstrap
def bca_ci(
    ids: list[str],
    actual: list[int],
    predicted: list[int],
    n_classes: int,
    metric: METRICS_LITERAL,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int | None = None,
) -> tuple[float, float, float] | tuple[list[float], list[float], list[float]]:
    cm_ps = _create_per_subject_confusion_matrix(ids=ids, actual=actual, predicted=predicted, n_classes=n_classes)

    point = METRIC_MAP[metric](cm_ps.sum(axis=0), strict=False)
    boot = _bootstrap(subject_confusion_matrices=cm_ps, metric=metric, n_resamples=n_resamples, seed=seed)
    jack = _jackknife(subject_confusion_matrices=cm_ps, metric=metric)

    point = np.atleast_1d(point)
    boot = boot if boot.ndim == 2 else boot.reshape(boot.shape[0], 1)
    jack = jack if jack.ndim == 2 else jack.reshape(jack.shape[0], 1)

    lo, hi = _bca_vector(point=point, boot=boot, jack=jack, confidence_level=confidence_level)

    if is_class_metric(metric):
        return point.tolist(), lo.tolist(), hi.tolist()
    else:
        return float(point[0]), float(lo[0]), float(hi[0])


def bca_ci_diff(
    ids: list[str],
    actual: list[int],
    predicted_a: list[int],
    predicted_b: list[int],
    n_classes: int,
    metric: METRICS_LITERAL,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int | None = None,
) -> tuple[float, float, float] | tuple[list[float], list[float], list[float]]:
    cm_a = _create_per_subject_confusion_matrix(ids, actual, predicted_a, n_classes)
    cm_b = _create_per_subject_confusion_matrix(ids, actual, predicted_b, n_classes)

    point_a = METRIC_MAP[metric](cm_a.sum(axis=0), strict=False)
    point_b = METRIC_MAP[metric](cm_b.sum(axis=0), strict=False)
    point = np.atleast_1d(point_a) - np.atleast_1d(point_b)

    boot_a = _bootstrap(cm_a, metric=metric, n_resamples=n_resamples, seed=seed)
    boot_b = _bootstrap(cm_b, metric=metric, n_resamples=n_resamples, seed=seed)
    if boot_a.ndim == 1:
        boot_a = boot_a.reshape(boot_a.shape[0], 1)
        boot_b = boot_b.reshape(boot_b.shape[0], 1)
    boot = boot_a - boot_b

    jack_a = _jackknife(cm_a, metric=metric)
    jack_b = _jackknife(cm_b, metric=metric)
    if jack_a.ndim == 1:
        jack_a = jack_a.reshape(jack_a.shape[0], 1)
        jack_b = jack_b.reshape(jack_b.shape[0], 1)
    jack = jack_a - jack_b

    lo, hi = _bca_vector(point=point, boot=boot, jack=jack, confidence_level=confidence_level)

    if is_class_metric(metric):
        return point.tolist(), lo.tolist(), hi.tolist()
    else:
        return float(point[0]), float(lo[0]), float(hi[0])


# helper
def is_class_metric(metric: METRICS_LITERAL) -> bool:
    return metric in ["precision", "recall", "f1_score"]


def _sanity_check(actual: list[int], predicted: list[int], n_classes: int) -> None:
    if len(actual) != len(predicted):
        raise ValueError("The length of `actual` and `predicted` must be the same.")

    if n_classes < 2:
        raise ValueError("There must be at least 2 classes.")

    valid_classes = set(range(n_classes))
    invalid_classes = set(actual) - valid_classes

    if invalid_classes:
        raise ValueError(f"`actual` contains invalid classes: {sorted(invalid_classes)}")

    invalid_predicted = set(predicted) - valid_classes

    if invalid_predicted:
        raise ValueError(f"`predicted` contains invalid classes: {sorted(invalid_predicted)}")


def _create_per_subject_confusion_matrix(
    ids: list[str], actual: list[int], predicted: list[int], n_classes: int
) -> NDArray[np.int64]:
    _sanity_check(actual=actual, predicted=predicted, n_classes=n_classes)

    if not (len(ids) == len(actual) == len(predicted)):
        raise ValueError("The length of `ids`, `actual` and `predicted` must be the same.")

    ids_ = np.asarray(ids)
    actual_ = np.asarray(actual, dtype=np.int64)
    predicted_ = np.asarray(predicted, dtype=np.int64)
    uniq, inv = np.unique(ids_, return_inverse=True)
    n_ids = uniq.size

    cm_ps = np.zeros((n_ids, n_classes, n_classes), dtype=np.int64)
    np.add.at(cm_ps, (inv, actual_, predicted_), 1)

    return cm_ps


def _accuracy_from_confusion(cm: NDArray[np.integer], strict: bool = True) -> NDArray[np.float64]:
    total = _total_from_confusion(cm)
    if np.any(total == 0):
        if cm.ndim > 2:
            raise ValueError("Accuracy is undefined, at least one confusion matrix contains no samples.")
        else:
            raise ValueError("Accuracy is undefined, the confusion matrix contains no samples.")

    correct = _correct_from_confusion(cm)
    return np.divide(correct, total, out=np.zeros_like(correct, dtype=np.float64))


def _balanced_accuracy_from_confusion(cm: NDArray[np.integer], strict: bool = True) -> NDArray[np.float64]:
    return _recall_from_confusion(cm=cm, strict=strict).mean(axis=-1)


def _macro_f1_score_from_confusion(cm: NDArray[np.integer], strict: bool = True) -> NDArray[np.float64]:
    return _f1_score_from_confusion(cm=cm, strict=strict).mean(axis=-1)


def _weighted_f1_score_from_confusion(cm: NDArray[np.integer], strict: bool = True) -> NDArray[np.float64]:
    support = _support_from_confusion(cm)
    total = support.sum(axis=-1)
    if strict and np.any(total == 0):
        if cm.ndim > 2:
            raise ValueError("Weighted F1 score is undefined, at least one confusion matrix contains no samples.")
        else:
            raise ValueError("Weighted F1 score is undefined, the confusion matrix contains no samples.")

    f1 = _f1_score_from_confusion(cm=cm, strict=strict)
    return np.divide((f1 * support).sum(axis=-1), total, out=np.zeros_like(total, dtype=np.float64), where=total > 0)


def _precision_from_confusion(cm: NDArray[np.integer], strict: bool = True) -> NDArray[np.float64]:
    predicted = _predicted_from_confusion(cm)

    if strict and np.any(predicted == 0):
        if cm.ndim > 2:
            raise ValueError("Precision is undefined, at least one class in one confusion matrix is never predicted.")
        else:
            raise ValueError("Precision is undefined, at least one class is never predicted.")

    tp = _tp_from_confusion(cm)
    return np.divide(tp, predicted, out=np.zeros_like(tp, dtype=np.float64), where=predicted > 0)


def _recall_from_confusion(cm: NDArray[np.integer], strict: bool = True) -> NDArray[np.float64]:
    support = _support_from_confusion(cm)

    if strict and np.any(support == 0):
        if cm.ndim > 2:
            raise ValueError("Recall is undefined, at least one class in one confusion matrix has no support.")
        else:
            raise ValueError("Recall is undefined, at least one class has no support.")

    tp = _tp_from_confusion(cm)
    return np.divide(tp, support, out=np.zeros_like(tp, dtype=np.float64), where=support > 0)


def _f1_score_from_confusion(cm: NDArray[np.integer], strict: bool = True) -> NDArray[np.float64]:
    prec = _precision_from_confusion(cm=cm, strict=strict)
    rec = _recall_from_confusion(cm=cm, strict=strict)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec, dtype=np.float64), where=(prec + rec) > 0)


def _support_from_confusion(cm: NDArray[np.integer]) -> NDArray[np.int64]:
    return cm.sum(axis=-1)


def _predicted_from_confusion(cm: NDArray[np.integer]) -> NDArray[np.int64]:
    return cm.sum(axis=-2)


def _total_from_confusion(cm: NDArray[np.integer]) -> NDArray[np.int64]:
    return cm.sum(axis=(-2, -1))


def _tp_from_confusion(cm: NDArray[np.integer]) -> NDArray[np.int64]:
    return np.diagonal(cm, axis1=-2, axis2=-1)


def _correct_from_confusion(cm: NDArray[np.integer]) -> NDArray[np.int64]:
    return np.trace(cm, axis1=-2, axis2=-1)


def _bootstrap(
    subject_confusion_matrices: NDArray[np.int64],
    metric: METRICS_LITERAL,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int | None = None,
) -> NDArray[np.float64]:
    cm_ps = subject_confusion_matrices
    n_ids = cm_ps.shape[0]
    rng = np.random.default_rng(seed=seed if seed is not None else system.get_seed())
    idxs = rng.integers(low=0, high=n_ids, size=(n_resamples, n_ids))

    result = np.empty((n_resamples, cm_ps.shape[1]) if is_class_metric(metric) else n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        pooled = cm_ps[idxs[i]].sum(axis=0)
        result[i] = METRIC_MAP[metric](pooled, strict=False)

    return result


def _jackknife(
    subject_confusion_matrices: NDArray[np.int64],
    metric: METRICS_LITERAL,
) -> NDArray[np.float64]:
    cm_ps = subject_confusion_matrices
    n_ids = cm_ps.shape[0]

    total = cm_ps.sum(axis=0)
    result = np.empty((n_ids, cm_ps.shape[1]) if is_class_metric(metric) else n_ids, dtype=np.float64)

    for i in range(n_ids):
        pooled_minus_i = total - cm_ps[i]
        result[i] = METRIC_MAP[metric](pooled_minus_i, strict=False)

    return result


def _bca_vector(
    point: NDArray[np.float64],
    boot: NDArray[np.float64],
    jack: NDArray[np.float64],
    confidence_level: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_resamples = boot.shape[0]
    n_samples = jack.shape[0]

    if n_resamples < 10:
        raise ValueError("Need at least ~10 bootstrap samples for a CI.")
    if n_samples < 2:
        raise ValueError("Need at least 2 jackknife values.")

    deg = np.all(np.isclose(boot, boot[0:1, :], equal_nan=True), axis=0)

    alpha = 1.0 - confidence_level
    zl = stats.norm.ppf(alpha / 2.0)
    zu = stats.norm.ppf(1.0 - alpha / 2.0)

    less = np.sum(boot < point[None, :], axis=0)
    ties = np.sum(boot == point[None, :], axis=0)
    phat = (less + 0.5 * ties) / float(n_resamples)
    eps = 1.0 / (n_resamples + 1.0)
    phat = np.clip(phat, eps, 1.0 - eps)
    z0 = stats.norm.ppf(phat)

    tbar = jack.mean(axis=0)
    u = tbar[None, :] - jack
    denom = 6.0 * (np.sum(u**2, axis=0) ** 1.5)

    num = np.sum(u**3, axis=0)
    a = np.divide(num, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom != 0.0)

    def q_adj(z: float) -> float:
        return stats.norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))

    ql = np.clip(q_adj(zl), eps, 1.0 - eps)
    qu = np.clip(q_adj(zu), eps, 1.0 - eps)

    n_classes = point.shape[0]
    lo = np.empty(n_classes, dtype=np.float64)
    hi = np.empty(n_classes, dtype=np.float64)
    for c in range(n_classes):
        if deg[c]:
            v = float(boot[0, c])
            lo[c] = v
            hi[c] = v
        else:
            lo[c] = float(np.quantile(boot[:, c], float(ql[c]), method="linear"))
            hi[c] = float(np.quantile(boot[:, c], float(qu[c]), method="linear"))
    return lo, hi


METRIC_MAP = {
    "accuracy": _accuracy_from_confusion,
    "balanced_accuracy": _balanced_accuracy_from_confusion,
    "macro_f1_score": _macro_f1_score_from_confusion,
    "weighted_f1_score": _weighted_f1_score_from_confusion,
    "precision": _precision_from_confusion,
    "recall": _recall_from_confusion,
    "f1_score": _f1_score_from_confusion,
}
