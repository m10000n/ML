from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar, Literal, cast

import nibabel as nib
import numpy as np
import safetensors.torch as st
import torch
from nibabel.nifti1 import Nifti1Image
from tqdm import tqdm

from data.HCP_1200.hcp_1200_metadata import (
    AVG_DURATION,
    AVG_REPS,
    MAX_REPS,
    SUBJECTS,
    TASKS,
)
from helper import aws, file, input, lock, path, system, time
from helper.class_ import Data_, DataConfig_
from helper.helper_ import round_to_str
from helper.print import print_end, print_error, print_start
from helper.validator import validator

##### config start #####
DIR: Path = path.home() / "HCP_1200"
##### config end #####

_TASKS = dict[str, list[str]]
_EVENT = list[tuple[float, float]]
_MODE = Literal["preprocessed", "unprocessed"]
_PE = Literal["LR", "RL"]


class HCP1200Data(Data_):
    DATA_LOCK = lock.get_lock("HCP_1200")

    ENV_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]

    REGION_NAME = "us-east-1"
    BUCKET_NAME = "hcp-openaccess"
    STUDY_PREFIX = "HCP_1200"

    TR = 0.72
    DELTA = 8
    DELTA_VOLUMES = math.ceil(DELTA / 0.72)

    SHAPE_UNPROCESSED = (90, 104, 72)
    SHAPE_PREPROCESSED = (75, 93, 81)

    BRAIN_BOUNDARIES = {
        "x_start": 8,
        "x_end": 83,
        "y_start": 8,
        "y_end": 101,
        "z_start": 5,
        "z_end": 86,
    }

    _TASK_DIR_ = str(DIR) + "/{mode}/{pe}/{subject}/{task}"
    _MRI_DIR_ = _TASK_DIR_ + "/{subtask}"
    _MRI_PATH_ = _MRI_DIR_ + "/{mode}_{pe}_{subject}_{task}_{subtask}_{n}.safetensors"
    _MRI_PATH_TEMP_ = _TASK_DIR_ + "/temp.nii.gz"

    _EXCLUDED_PATH = DIR / "excluded.json"
    _EVENT_PATH = DIR / "event.json"
    _METADATA_PATH = DIR / "metadata.txt"

    _EVENT_LOCK = lock.get_lock("event")
    _EXCLUDED_LOCK = lock.get_lock("excluded")

    _cached_events = None
    _cached_excluded = None

    config: HCP1200DataConfig

    def __init__(self, config: HCP1200DataConfig):
        super().__init__(config)
        self._downloaded = False
        self._metadata: list[str] | None = None

    # subjects
    @staticmethod
    def get_subjects(subjects: float | int | list[str]) -> list[str]:
        sorted_subjects = sorted(SUBJECTS)
        if isinstance(subjects, float):
            return sorted_subjects[: int(len(sorted_subjects) * subjects)]
        elif isinstance(subjects, int):
            return sorted_subjects[:subjects]
        else:
            invalid_subjects = [id for id in subjects if id not in SUBJECTS]
            if invalid_subjects:
                raise ValueError(f"Invalid subject(s): {", ".join(invalid_subjects)}.")
            return subjects

    @staticmethod
    def calculate_subjects_fraction(subjects: int | list[str]) -> float:
        n_subjects = len(subjects) if isinstance(subjects, list) else subjects
        return n_subjects / len(SUBJECTS)

    # remote paths
    @staticmethod
    def _get_task_dir_r(mode: _MODE, pe: _PE, subject: str, task: str) -> str:
        mode_ = "unprocessed/3T" if mode == "unprocessed" else "MNINonLinear/Results"
        return f"HCP_1200/{subject}/{mode_}/tfMRI_{task}_{pe}"

    @staticmethod
    def _get_mri_path_r(
        mode: _MODE,
        pe: _PE,
        subject: str,
        task: str,
    ) -> str:
        file_name = f"{subject}_3T_tfMRI_{task}_{pe}.nii.gz" if mode == "unprocessed" else f"tfMRI_{task}_{pe}.nii.gz"
        return HCP1200Data._get_task_dir_r(mode=mode, pe=pe, subject=subject, task=task) + "/" + file_name

    @staticmethod
    def _get_event_path_r(pe: _PE, subject: str, task: str, subtask: str) -> str:
        return (
            f"{HCP1200Data._get_task_dir_r(mode="preprocessed", pe=pe, subject=subject, task=task)}/EVs/{subtask}.txt"
        )

    # events
    @staticmethod
    def _get_event(pe: _PE, subject: str, task: str, subtask: str) -> _EVENT | str:
        try:
            return HCP1200Data._get_all_events()[pe][subject][task][subtask]
        except KeyError:
            raise FileNotFoundError(
                dedent(
                    f"""\
                    Event file not found.
                        pe: {pe}
                        subject: {subject}
                        task: {task}
                        subtask: {subtask}
                        """
                )
            )

    @staticmethod
    def _get_missing_events(
        pes: list[_PE], subjects: list[str], tasks: _TASKS
    ) -> tuple[list[tuple[_PE, str, str, str]], int, int]:  # [(pe, subject, task, subtask), ...], n_missing, n_total

        missing: list = []
        n_missing = 0
        n_total = 0

        for pe in pes:
            for subject in subjects:
                for task in tasks:
                    for subtask in tasks[task]:
                        if not HCP1200Data._event_downloaded(subject=subject, pe=pe, task=task, subtask=subtask):
                            missing.append((pe, subject, task, subtask))
                            n_missing += 1
                        n_total += 1

        return missing, n_missing, n_total

    @staticmethod
    def _get_all_events() -> dict:
        with HCP1200Data._EVENT_LOCK:
            if HCP1200Data._cached_events is None:

                if HCP1200Data._EVENT_PATH.exists():
                    HCP1200Data._cached_events = file.read_json(path=HCP1200Data._EVENT_PATH, unlock=True)
                else:
                    HCP1200Data._cached_events = {}

            return HCP1200Data._cached_events

    @staticmethod
    def _event_downloaded(pe: _PE, subject: str, task: str, subtask: str) -> bool:
        try:
            HCP1200Data._get_event(pe=pe, subject=subject, task=task, subtask=subtask)
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def _add_event(
        pe: _PE,
        subject: str,
        task: str,
        subtask: str,
        events: _EVENT | str,
    ) -> None:
        with HCP1200Data._EVENT_LOCK:
            HCP1200Data._get_all_events().setdefault(pe, {}).setdefault(subject, {}).setdefault(task, {}).setdefault(
                subtask, events
            )

    @staticmethod
    def _download_event(
        args: tuple[
            _PE,  # pe
            str,  # subject
            str,  # task
            str,  # subtask
        ],
    ) -> tuple[_PE, str, str, str, _EVENT | str]:  # pe, subject, task, subtask, event
        pe, subject, task, subtask = args

        remote_path = HCP1200Data._get_event_path_r(pe=pe, subject=subject, task=task, subtask=subtask)

        try:
            client = aws.get_client(region_name=HCP1200Data.REGION_NAME)
            event = aws.get_file(client=client, bucket=HCP1200Data.BUCKET_NAME, file_path=remote_path).decode("utf-8")
            if not event:
                raise ValueError("Event file is empty.")
            lines = event.strip().split("\n")
            event_: _EVENT | str = [
                (float(onset), float(duration)) for line in lines for onset, duration in [line.split("\t")[:2]]
            ]
        except (FileNotFoundError, ValueError):
            event_ = "MISSING"

        return pe, subject, task, subtask, event_

    @staticmethod
    def _flush_events() -> None:
        with HCP1200Data._EVENT_LOCK:
            file.write_json(path=HCP1200Data._EVENT_PATH, data=HCP1200Data._get_all_events(), overwrite=True, lock=True)
            HCP1200Data._cached_events = None

    # excluded
    @staticmethod
    def _get_all_excluded() -> dict:
        with HCP1200Data._EXCLUDED_LOCK:
            if HCP1200Data._cached_excluded is None:
                if HCP1200Data._EXCLUDED_PATH.exists():
                    HCP1200Data._cached_excluded = file.read_json(path=HCP1200Data._EXCLUDED_PATH, unlock=True)
                else:
                    HCP1200Data._cached_excluded = {}

            return HCP1200Data._cached_excluded

    @staticmethod
    def _is_excluded(mode: _MODE, pe: _PE, subject: str, task: str, subtask: str, n: int) -> bool:
        subtask_excluded = (
            HCP1200Data._get_all_excluded().get(mode, {}).get(pe, {}).get(subject, {}).get(task, {}).get(subtask, {})
        )
        if subtask_excluded and str(n) in subtask_excluded:
            return True
        return False

    @staticmethod
    def _add_excluded(mode: _MODE, pe: _PE, subject: str, task: str, subtask: str, n: int, n_volumes: int) -> None:
        with HCP1200Data._EXCLUDED_LOCK:
            HCP1200Data._get_all_excluded().setdefault(mode, {}).setdefault(pe, {}).setdefault(subject, {}).setdefault(
                task, {}
            ).setdefault(subtask, {})[n] = n_volumes

    @staticmethod
    def _flush_excluded() -> None:
        with HCP1200Data._EXCLUDED_LOCK:
            file.write_json(
                path=HCP1200Data._EXCLUDED_PATH, data=HCP1200Data._get_all_excluded(), overwrite=True, lock=True
            )
            HCP1200Data._cached_excluded = None

    # MRI
    @staticmethod
    def _get_missing_mri(
        mode: _MODE,
        pes: list[_PE],
        subjects: list[str],
        tasks: _TASKS,
        subtask_limit: int | None,
    ) -> tuple[
        list[tuple[_MODE, _PE, str, str, dict[str, _EVENT]]], int, int
    ]:  # [(mode, pe, subject, task, subtasks), ...], n_missing, n_total
        missing = []
        n_missing = 0
        n_total = 0

        for pe in pes:
            for subject in subjects:
                for task in tasks:
                    missing_: tuple = (mode, pe, subject, task, {})
                    mri_needed = False

                    for subtask in tasks[task]:
                        subtask_events = HCP1200Data._get_event(subject=subject, pe=pe, task=task, subtask=subtask)

                        if subtask_events == "MISSING":
                            continue

                        max_n = min(subtask_limit, len(subtask_events)) if subtask_limit else len(subtask_events)

                        for n in range(max_n):
                            if not HCP1200Data._is_excluded(
                                mode=mode,
                                pe=pe,
                                subject=subject,
                                task=task,
                                subtask=subtask,
                                n=n,
                            ):
                                mri_needed = True
                                if not Path(
                                    HCP1200Data._MRI_PATH_.format(
                                        subject=subject,
                                        pe=pe,
                                        task=task,
                                        subtask=subtask,
                                        mode=mode,
                                        n=n,
                                    )
                                ).exists():
                                    missing_[4][subtask] = subtask_events
                                    missing.append(missing_)
                                    n_missing += 1
                                    break

                    if mri_needed:
                        n_total += 1

        return missing, n_missing, n_total

    @staticmethod
    def _download_mri(
        args: tuple[
            _MODE,  # mode
            _PE,  # pe
            str,  # subject
            str,  # task
            dict[str, _EVENT],  # subtasks
            int | None,  # subtask_limit
            int | None,  # min_volumes
        ],
    ) -> list[
        tuple[
            _MODE,  # mode
            _PE,  # pe
            str,  # subject
            str,  # task
            str,  # subtask
            int,  # n
            int,  # n_volumes
        ]
    ]:
        mode, pe, subject, task, subtasks, subtask_limit, min_volumes = args

        remote_path = HCP1200Data._get_mri_path_r(mode=mode, pe=pe, subject=subject, task=task)
        temp_path = Path(HCP1200Data._MRI_PATH_TEMP_.format(mode=mode, pe=pe, subject=subject, task=task))
        task_dir = Path(HCP1200Data._TASK_DIR_.format(mode=mode, pe=pe, subject=subject, task=task))
        mri_dirs = [
            Path(HCP1200Data._MRI_DIR_.format(mode=mode, pe=pe, subject=subject, task=task, subtask=subtask))
            for subtask in subtasks
        ]

        try:
            os.makedirs(task_dir, exist_ok=True)

            client = aws.get_client(region_name=HCP1200Data.REGION_NAME)

            try:
                aws.download(
                    client=client,
                    bucket=HCP1200Data.BUCKET_NAME,
                    file_path=remote_path,
                    local_file_path=temp_path,
                )
            except FileNotFoundError:
                excluded = []
                for subtask, subtask_events in subtasks.items():
                    if subtask_limit is None or subtask_limit > MAX_REPS[subtask]:
                        subtask_limit = MAX_REPS[subtask]

                    for n in range(min(subtask_limit, len(subtask_events))):
                        excluded.append((mode, pe, subject, task, subtask, n, 0))

                return excluded

            if mode == "preprocessed":
                dtype: type[np.int16] | type[np.float32] = np.float32
            else:
                dtype = np.int16

            mri_file = cast(Nifti1Image, nib.load(filename=temp_path))
            mri = np.array(mri_file.dataobj, dtype=dtype)

            if mode == "preprocessed":
                mri = mri[
                    HCP1200Data.BRAIN_BOUNDARIES["x_start"] : HCP1200Data.BRAIN_BOUNDARIES["x_end"],
                    HCP1200Data.BRAIN_BOUNDARIES["y_start"] : HCP1200Data.BRAIN_BOUNDARIES["y_end"],
                    HCP1200Data.BRAIN_BOUNDARIES["z_start"] : HCP1200Data.BRAIN_BOUNDARIES["z_end"],
                ]

            excluded = []

            for (subtask, subtask_events), mri_dir in zip(subtasks.items(), mri_dirs):
                os.makedirs(mri_dir, exist_ok=True)

                if subtask_limit is None or subtask_limit > MAX_REPS[subtask]:
                    subtask_limit = MAX_REPS[subtask]

                for n, (onset, duration) in enumerate(subtask_events):
                    if n >= subtask_limit:
                        break

                    start = math.floor(onset / HCP1200Data.TR)
                    end = math.ceil((onset + duration + HCP1200Data.DELTA) / HCP1200Data.TR)
                    mri_ = mri[..., start : (end + 1)]

                    n_volumes = mri_.shape[-1]
                    if (end + 1) - start > n_volumes or min_volumes and min_volumes > n_volumes:
                        excluded.append((mode, pe, subject, task, subtask, n, n_volumes))
                        continue

                    mri_torch = torch.from_numpy(mri_).permute(3, 0, 1, 2)

                    mean = mri_torch.mean() if mode == "preprocessed" else mri_torch.to(torch.float32).mean()
                    std = mri_torch.std() if mode == "preprocessed" else mri_torch.to(torch.float32).std()
                    max = mri_torch.max() if mode == "preprocessed" else mri_torch.to(torch.float32).max()

                    data = {"tensor": mri_torch.clone().contiguous(), "mean": mean, "std": std, "max": max}

                    if mode == "preprocessed":
                        data["mean_voxel"] = mri_torch.mean(dim=0)
                        data["std_voxel"] = mri_torch.std(dim=0)
                        data["min_voxel"] = mri_torch.amin(dim=0)
                        data["max_voxel"] = mri_torch.amax(dim=0)

                    st.save_file(
                        tensors=data,
                        filename=HCP1200Data._MRI_PATH_.format(
                            mode=mode,
                            pe=pe,
                            subject=subject,
                            task=task,
                            subtask=subtask,
                            n=n,
                        ),
                    )

            return excluded
        except Exception as e:
            raise RuntimeError(
                dedent(
                    f"""\
                    Failed to download fMRI file.
                        mode: {mode}
                        pe: {pe}
                        subject: {subject}
                        task: {task}
                        subtasks: {subtasks}      
                        subtask limit: {subtask_limit}
                        min volumes: {min_volumes}
                        """
                )
            ) from e
        finally:
            if temp_path.exists():
                temp_path.unlink()

            if task_dir.exists() and path.is_empty(task_dir):
                task_dir.rmdir()

            for mri_dir in mri_dirs:
                if mri_dir.exists() and path.is_empty(mri_dir):
                    mri_dir.rmdir()

    # paths and labels
    def get_data_label_id(
        self,
        subjects: list[str],
    ) -> list[tuple[Path, tuple[str, str], str]]:  # [(path, (task, subtask), id), ...]
        if not self._downloaded:
            raise ValueError("The dataset must be downloaded before calling `get_data_label`.")

        result = []

        for pe in self.config.pes:
            for subject in subjects:
                for task in self.config.tasks:
                    for subtask in self.config.tasks[task]:
                        if not self.config.subtask_limit or self.config.subtask_limit > MAX_REPS[subtask]:
                            subtask_limit_ = MAX_REPS[subtask]

                        for n in range(subtask_limit_):
                            path = Path(
                                self._MRI_PATH_.format(
                                    mode=self.config.mode,
                                    pe=pe,
                                    subject=subject,
                                    task=task,
                                    subtask=subtask,
                                    n=n,
                                )
                            )
                            if path.exists():
                                result.append((path, (task, subtask), subject))
                            else:
                                break

        return result

    @validator.constraints("subtask_limit", "x > 0")
    @validator.constraints("min_volumes", "x > 0")
    def get_included_subjects(self, subtask_limit: int | None = None, min_volumes: int | None = None) -> list[str]:
        if not self.is_downloaded():
            raise ValueError("The dataset must be downloaded before calling `get_included_subjects`.")

        if min_volumes is None:
            if self.config.min_volumes is None:
                min_volumes_ = 1
            else:
                min_volumes_ = self.config.min_volumes
        else:
            min_volumes_ = min_volumes

        all_evs = self._get_all_events()
        all_excluded = self._get_all_excluded()

        excluded_subjects = []

        for subject in self.config.ids:
            is_included = False
            for pe in self.config.pes:
                if is_included:
                    break
                for task in self.config.tasks:
                    if is_included:
                        break
                    for subtask in self.config.tasks[task]:
                        if is_included:
                            break
                        ev = all_evs.get(pe, {}).get(subject, {}).get(task, {}).get(subtask, [])
                        excluded = (
                            all_excluded.get(self.config.mode, {})
                            .get(pe, {})
                            .get(subject, {})
                            .get(task, {})
                            .get(subtask, {})
                        )

                        if subtask_limit is None:
                            if self.config.subtask_limit is None or self.config.subtask_limit > MAX_REPS[subtask]:
                                subtask_limit_ = MAX_REPS[subtask]
                            else:
                                subtask_limit_ = self.config.subtask_limit
                        else:
                            if subtask_limit > MAX_REPS[subtask]:
                                subtask_limit_ = MAX_REPS[subtask]
                            else:
                                subtask_limit_ = subtask_limit

                        if ev != "MISSING":
                            for ev_ in ev:
                                if int(ev_[1] / HCP1200Data.TR) >= min_volumes_:
                                    if not excluded:
                                        is_included = True
                                        break
                                    else:
                                        for i in range(subtask_limit_):
                                            if i not in excluded or excluded[i] >= min_volumes_:
                                                is_included = True
                                                break

            if not is_included:
                excluded_subjects.append(subject)

        return excluded_subjects

    # download
    def is_downloaded(self) -> bool:
        return self._downloaded

    def download(
        self,
    ) -> None:
        with self.DATA_LOCK:
            num_workers = system.get_num_workers(download=True)

            try:
                print_start(
                    text=f"Start downloading HCP 1200 dataset. | {time.now_str()} | Number of Workers: {num_workers}",
                    mode="primary",
                )

                missing_events, n_missing_events, n_total_events = self._get_missing_events(
                    subjects=self.config.ids, pes=self.config.pes, tasks=self.config.tasks
                )
                if n_missing_events > 0:
                    try:
                        input.env_vars_exist(HCP1200Data.ENV_VARS)
                    except ValueError as e:
                        print_error(e.args[0])
                        sys.exit(1)

                    progress_bar = tqdm(
                        total=n_total_events,
                        initial=n_total_events - n_missing_events,
                        desc="Downloading event files",
                    )

                    with Pool(processes=num_workers) as pool:
                        for pe, subject, task, subtask, events in pool.imap_unordered(
                            HCP1200Data._download_event, missing_events
                        ):
                            self._add_event(pe=pe, subject=subject, task=task, subtask=subtask, events=events)
                            progress_bar.update(1)

                    progress_bar.close()
                    self._flush_events()

                missing_mri, n_missing_mri, n_total_mri = self._get_missing_mri(
                    mode=self.config.mode,
                    pes=self.config.pes,
                    subjects=self.config.ids,
                    tasks=self.config.tasks,
                    subtask_limit=self.config.subtask_limit,
                )

                if n_missing_mri > 0:
                    try:
                        input.env_vars_exist(HCP1200Data.ENV_VARS)
                    except ValueError as e:
                        print_error(e.args[0])
                        sys.exit(1)

                    progress_bar = tqdm(
                        total=n_total_mri,
                        initial=n_total_mri - n_missing_mri,
                        desc="Downloading fMRI files",
                    )

                    args_list = [
                        (
                            *missing_,
                            self.config.subtask_limit,
                            self.config.min_volumes,
                        )
                        for missing_ in missing_mri
                    ]

                    with Pool(processes=num_workers) as pool:
                        for excluded in pool.imap_unordered(HCP1200Data._download_mri, args_list):
                            for excluded_ in excluded:
                                self._add_excluded(*excluded_)

                            progress_bar.update(1)

                    progress_bar.close()

                self._downloaded = True
                self._write_metadata()

                print_end(text="Finished downloading dataset.", mode="primary")
            except BaseException as e:
                print_error(text="Failed to download dataset.", mode="primary")
                raise e
            finally:
                self._flush_events()
                self._flush_excluded()

    # estimate dataset size
    def size_info(self) -> None:
        mri_size = (
            math.prod(self.SHAPE_PREPROCESSED) * 4
            if self.config.mode == "preprocessed"
            else math.prod(self.SHAPE_UNPROCESSED) * 2
        )

        total_size = 0.0
        for task in self.config.tasks:
            for subtask in self.config.tasks[task]:
                volumes = math.ceil((AVG_DURATION[subtask] + self.DELTA) / self.TR)
                subtask_limit_ = (
                    self.config.subtask_limit
                    if self.config.subtask_limit and self.config.subtask_limit < AVG_REPS[subtask]
                    else AVG_REPS[subtask]
                )
                total_size += mri_size * volumes * subtask_limit_

        total_size *= len(self.config.pes) * len(self.config.ids)

        print(
            dedent(
                f"""\
            Estimated dataset size: {round_to_str(x=total_size / 1024**3, digits=3)} GB.
            
            Note: This estimate is approximate and is derived from the average duration and frequency of the subtasks.
            Actual disk usage will vary!"""
            )
        )

    # class distribution and frequency
    def get_class_distribution(
        self, subjects: list[str] | None = None
    ) -> dict[Literal["task", "subtask"], dict[str, int]]:
        if not self._downloaded:
            raise ValueError("The dataset must be downloaded before calling `get_class_distribution`.")

        subjects_ = subjects if subjects else self.config.ids
        classes = self.config.get_classes()

        tasks = {task: 0 for task in classes["task"]}
        subtasks = {subtask: 0 for subtask in classes["subtask"]}

        for pe in self.config.pes:
            for subject in subjects_:
                for task in self.config.tasks:
                    for subtask in self.config.tasks[task]:
                        max_n = self.config.subtask_limit if self.config.subtask_limit else MAX_REPS[subtask]
                        for n in range(max_n):
                            mri_path = Path(
                                self._MRI_PATH_.format(
                                    mode=self.config.mode, pe=pe, subject=subject, task=task, subtask=subtask, n=n
                                )
                            )
                            if mri_path.exists():
                                tasks[task] += 1
                                subtasks[subtask] += 1

        return {"task": tasks, "subtask": subtasks}

    # metadata
    def get_metadata(self) -> list[str]:
        if not self._downloaded:
            raise ValueError("The dataset must be downloaded before calling `get_metadata`.")

        if self._metadata:
            return self._metadata

        stats: dict[str, dict[str, Any]] = {
            "pes": defaultdict(lambda: {"included": 0, "excluded": 0}),
            "subjects": {"included": set(), "total": len(self.config.ids)},
            "tasks": defaultdict(lambda: {"included": 0, "excluded": 0}),
            "subtasks": defaultdict(lambda: defaultdict(lambda: {"included": 0, "excluded": 0})),
            "size": {"total": 0},
        }

        for pe in self.config.pes:
            for subject in self.config.ids:
                for task in self.config.tasks:
                    for subtask in self.config.tasks[task]:
                        max_n = self.config.subtask_limit if self.config.subtask_limit else MAX_REPS[subtask]
                        for n in range(max_n):
                            mri_path = Path(
                                self._MRI_PATH_.format(
                                    mode=self.config.mode, pe=pe, subject=subject, task=task, subtask=subtask, n=n
                                )
                            )
                            if mri_path.exists():
                                stats["pes"][pe]["included"] += 1
                                stats["subjects"]["included"].add(subject)
                                stats["tasks"][task]["included"] += 1
                                stats["subtasks"][task][subtask]["included"] += 1
                                stats["size"]["total"] += mri_path.stat().st_size
                            elif HCP1200Data._is_excluded(
                                mode=self.config.mode, pe=pe, subject=subject, task=task, subtask=subtask, n=n
                            ):
                                stats["pes"][pe]["excluded"] += 1
                                stats["tasks"][task]["excluded"] += 1
                                stats["subtasks"][task][subtask]["excluded"] += 1
                            else:
                                break

        # Generate metadata text
        metadata = [
            f"Number of subjects: {len(stats['subjects']['included'])} ({stats['subjects']['total'] - len(stats['subjects']['included'])})",
            f"Number of all subtasks: {sum(t['included'] for t in stats['tasks'].values())} ({sum(t['excluded'] for t in stats['tasks'].values())})",
            f"Total file size: ~{round_to_str(x=stats['size']['total'] / (1024 ** 3), digits=3)} GB",
        ]

        # Add task statistics
        for task in self.config.tasks:
            task_stats = stats["tasks"][task]
            metadata.extend([f"\t{task}: {task_stats['included']} ({task_stats['excluded']})"])
            for subtask in self.config.tasks[task]:
                subtask_stats = stats["subtasks"][task][subtask]
                metadata.extend([f"\t\t{subtask}: {subtask_stats['included']} ({subtask_stats['excluded']})"])

        # Add phase encoding statistics
        metadata.extend(["Phase encodings:"])
        for pe in self.config.pes:
            pe_stats = stats["pes"][pe]
            metadata.append(f"\t{pe}: {pe_stats['included']} ({pe_stats['excluded']})")

        metadata.extend([f"Mode: {self.config.mode}", f"Subtask limit: {self.config.subtask_limit}"])

        self._metadata = metadata
        return metadata

    def _write_metadata(self) -> None:
        file.write_lines(path=self._METADATA_PATH, lines=self.get_metadata(), overwrite=True, lock=True)


@dataclass()
class HCP1200DataConfig(DataConfig_):
    class_: ClassVar[type[Data_]] = HCP1200Data
    name: ClassVar[str] = "HCP 1200"

    mode: _MODE
    pes: list[_PE]
    tasks: _TASKS
    subtask_limit: int | None
    min_volumes: int | None = 27

    @validator.constraints("subtask_limit", "x > 0")
    @validator.constraints("min_volumes", "x > 0")
    def __init__(
        self,
        description: str,
        subjects: float | int | list[str],
        mode: _MODE,
        pes: list[_PE],
        tasks: _TASKS,
        subtask_limit: int | None,
        min_volumes: int | None = 27,
    ) -> None:
        if isinstance(subjects, float):
            if not 0 <= subjects <= 1:
                raise ValueError(f"If `subjects` is a float, it must be in range [0, 1].")
        elif isinstance(subjects, int):
            if not 0 <= subjects <= len(SUBJECTS):
                raise ValueError(f"If `subjects` is an int, it must be in range [0, {len(SUBJECTS)}].")
        else:
            for subject in subjects:
                if subject not in SUBJECTS:
                    raise ValueError(f"Subject `{subject}` is invalid.")

        valid_tasks = TASKS.keys()
        for task, subtasks in tasks.items():
            if task not in valid_tasks:
                raise ValueError(f"Task `{task}` is invalid.")

            valid_subtasks = TASKS[task]
            for subtask in subtasks:
                if subtask not in valid_subtasks:
                    raise ValueError(f"Subtask `{subtask}` for task `{task}` is invalid.")

        if min_volumes is not None:
            if min_volumes < HCP1200Data.DELTA_VOLUMES:
                raise ValueError(
                    f"The minimum number of volumes ({min_volumes}) must be >= {HCP1200Data.DELTA_VOLUMES}."
                    f"If you want to work with a shorter hemodynamic response, change the constant `DELTA` in "
                    f"`{path.make_relative(path.file_path())}`. It is currently set to {HCP1200Data.DELTA} seconds."
                )

        subjects = HCP1200Data.get_subjects(subjects)
        super().__init__(description, ids=subjects)

        self.mode = mode
        self.pes = sorted(pes)
        self.tasks = {task: sorted(subtask) for task, subtask in sorted(tasks.items())}
        self.subtask_limit = subtask_limit
        self.min_volumes = min_volumes

    @staticmethod
    def from_dict(dict_: dict[str, Any]) -> HCP1200DataConfig:
        return HCP1200DataConfig(**dict_)

    def as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "mode": self.mode,
            "pes": self.pes,
            "subtask_limit": self.subtask_limit,
            "tasks": self.tasks,
        }

    def get_classes(self, pretty: bool = False) -> dict[Literal["task", "subtask"], list[str]]:
        return {
            "task": (
                [class_[0] + class_[1:].lower() if class_ != "WM" else class_ for class_ in self.tasks.keys()]
                if pretty
                else list(self.tasks.keys())
            ),
            "subtask": list(chain.from_iterable(self.tasks.values())),
        }
