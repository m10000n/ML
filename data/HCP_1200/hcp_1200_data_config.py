from typing import Literal

from data.HCP_1200.hcp_1200_data import HCP1200Data, HCP1200DataConfig
from helper.helper_ import round_to_str

# >>> Wang
_WANG_TASKS = {
    "EMOTION": ["fear"],
    "GAMBLING": ["loss"],
    "LANGUAGE": ["present_story"],
    "MOTOR": ["rh"],
    "RELATIONAL": ["relation"],
    "SOCIAL": ["mental"],
    "WM": ["2bk_places"],
}

_WANG_VOLUMES = 27


# preprocessed
def prep(
    subjects: float | int | list[str] = 1.0,
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"],
    subtask_limit: int | None = None,
    min_volumes: int = _WANG_VOLUMES,
) -> HCP1200DataConfig:
    if isinstance(subjects, float):
        subjects_fraction = subjects
    else:
        subjects_fraction = HCP1200Data.calculate_subjects_fraction(subjects)

    return HCP1200DataConfig(
        description=f"mode: preprocessed, subjects: {round_to_str(x=subjects_fraction, digits=2)}, pe: {"+".join(pes)}, subtask_limit: {subtask_limit}, min volumes: {min_volumes}",
        subjects=subjects,
        mode="preprocessed",
        pes=pes,
        tasks=_WANG_TASKS,
        subtask_limit=subtask_limit,
        min_volumes=min_volumes,
    )


def prep_10pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.1, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_20pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.2, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_30pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.3, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_40pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.4, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_50pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.5, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_60pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.6, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_70pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.7, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_80pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.8, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_90pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=0.9, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_10s(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=10, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def prep_50s(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return prep(subjects=50, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


# unprocessed
def unp(
    subjects: float | int | list[str] = 1.0,
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"],
    subtask_limit: int | None = None,
    min_volumes: int = _WANG_VOLUMES,
) -> HCP1200DataConfig:
    if isinstance(subjects, float):
        subjects_fraction = subjects
    else:
        subjects_fraction = HCP1200Data.calculate_subjects_fraction(subjects)

    return HCP1200DataConfig(
        description=f"mode: unprocessed, subjects: {round_to_str(x=subjects_fraction, digits=2)}, pe: {"+".join(pes)}, subtask_limit: {subtask_limit}, min volumes: {min_volumes}",
        subjects=subjects,
        mode="unprocessed",
        pes=pes,
        tasks=_WANG_TASKS,
        subtask_limit=subtask_limit,
        min_volumes=min_volumes,
    )


def unp_10pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.1, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_20pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.2, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_30pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.3, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_40pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.4, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_50pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.5, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_60pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.6, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_70pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.7, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_80pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.8, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_90pct(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=0.9, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_10s(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=10, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


def unp_50s(
    pes: list[Literal["LR", "RL"]] = ["LR", "RL"], subtask_limit: int | None = None, min_volumes: int = _WANG_VOLUMES
) -> HCP1200DataConfig:
    return unp(subjects=50, pes=pes, subtask_limit=subtask_limit, min_volumes=min_volumes)


# <<< Wang

# >>> complete
_ALL_TASKS = {
    "EMOTION": ["fear", "neut"],
    "GAMBLING": ["win", "loss", "win_event", "loss_event"],
    "LANGUAGE": [
        "story",
        "math",
        "present_math",
        "present_story",
        "question_math",
        "question_story",
        "response_math",
        "response_story",
    ],
    "MOTOR": ["cue", "lf", "rf", "lh", "rh", "t"],
    "RELATIONAL": ["relation", "math", "error"],
    "SOCIAL": ["mental", "rnd", "mental_resp", "other_resp"],
    "WM": [
        "0bk_body",
        "0bk_faces",
        "0bk_places",
        "0bk_tools",
        "2bk_body",
        "2bk_faces",
        "2bk_places",
        "2bk_tools",
        "0bk_cor",
        "0bk_err",
        "0bk_nlr",
        "2bk_cor",
        "2bk_err",
        "2bk_nlr",
        "all_bk_cor",
        "all_bk_err",
    ],
}


def prep_complete() -> HCP1200DataConfig:
    return HCP1200DataConfig(
        description="mode: preprocessed, min volumes: 27",
        subjects=1.0,
        mode="preprocessed",
        pes=["LR", "RL"],
        tasks=_ALL_TASKS,
        subtask_limit=None,
        min_volumes=_WANG_VOLUMES,
    )


def unp_complete() -> HCP1200DataConfig:
    return HCP1200DataConfig(
        description="mode: preprocessed, min volumes: 27",
        subjects=1.0,
        mode="unprocessed",
        pes=["LR", "RL"],
        tasks=_ALL_TASKS,
        subtask_limit=None,
        min_volumes=_WANG_VOLUMES,
    )


# <<< complete
