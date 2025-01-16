import math
import os
import random
import shutil
import sys
from multiprocessing import Manager, Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

import helper.aws as aws
import helper.json as json
from helper.folder import get_files, get_folders
from helper.system import get_num_physical_cores


# dataset size for TASK_LIMIT = 1: 660 GB
# dataset size for TASK_LIMIT = 4: ~1661GB
class TaskDataset(Dataset):
    SECRETE_NAME = "ConnectomeDB"
    REGION_NAME = "us-east-1"
    BUCKET_NAME = "hcp-openaccess"
    STUDY_PREFIX = "HCP_1200/"

    TR = 0.72
    DELTA = 8
    WINDOW_WIDTH = 27
    # max limit = 4
    TASK_LIMIT = 4
    BRAIN_BOUNDARIES = {
        "x_start": 8,
        "x_end": 83,
        "y_start": 8,
        "y_end": 101,
        "z_start": 5,
        "z_end": 86,
    }

    TASKS = ["EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM"]
    TASK_EVS = [
        "fear.txt",
        "loss.txt",
        "present_story.txt",
        "rh.txt",
        "relation.txt",
        "mental.txt",
        "2bk_places.txt",
    ]

    MRI_TASK_FORMAT = STUDY_PREFIX + "{id}/MNINonLinear/Results/tfMRI_{task}_LR/"
    MRI_FILE_FORMAT = MRI_TASK_FORMAT + "tfMRI_{task}_LR.nii.gz"
    EV_FILE_FORMAT = MRI_TASK_FORMAT + "EVs/{ev}"

    ROOT_DIR = Path(__file__).parent / "data"
    META_PATH = ROOT_DIR / "meta.json"

    def __init__(self, subject_ids, is_train=False):
        if not self.fully_downloaded():
            raise RuntimeError(
                "Make sure to download the dataset before creating an instance."
            )

        self.subject_ids = subject_ids
        self.is_train = is_train

        self.samples = []
        for subject_id in self.subject_ids:
            subject_path = self.ROOT_DIR / str(subject_id)
            for task_folder in subject_path.iterdir():
                for file in task_folder.glob("*.npy"):
                    self.samples.append((str(file), task_folder.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, task = self.samples[idx]
        data = torch.load(f=file_path, weights_only=True)

        if self.is_train:
            start_idx = np.random.randint(
                low=0, high=data.shape[0] - self.WINDOW_WIDTH + 1
            )
        else:
            start_idx = 0

        data = data[start_idx : start_idx + self.WINDOW_WIDTH, ...]
        data = data / torch.max(data)

        label = self.TASKS.index(task)

        return data, label

    @staticmethod
    def fully_downloaded():
        return TaskDataset.META_PATH.exists()

    @staticmethod
    def get_all_subject_ids():
        if not TaskDataset.fully_downloaded():
            RuntimeError(
                "This function must not be called before the dataset is fully downloaded."
            )
        return [folder.name for folder in get_folders(TaskDataset.ROOT_DIR)]

    @staticmethod
    def download_dataset(num_workers=4):
        os.makedirs(TaskDataset.ROOT_DIR, exist_ok=True)

        secrete = aws.get_secret(
            secrete_name=TaskDataset.SECRETE_NAME, region_name=TaskDataset.REGION_NAME
        )
        client = aws.get_client(secrete=secrete)

        _, subject_paths = aws.get_content(
            client=client,
            bucket=TaskDataset.BUCKET_NAME,
            folder_path=TaskDataset.STUDY_PREFIX,
            verbose=False,
        )
        subject_ids = [subject_path.split("/")[-2] for subject_path in subject_paths]
        downloaded_subject_ids = [
            folder.name for folder in get_folders(TaskDataset.ROOT_DIR)
        ]
        required_subject_ids = [
            subject_id
            for subject_id in subject_ids
            if subject_id not in downloaded_subject_ids
        ]

        manager = Manager()
        shared_progress = manager.dict(
            downloaded=len(downloaded_subject_ids), total=len(subject_ids)
        )
        excluded_data = manager.dict()
        excluded_data["path"] = manager.list()

        temp_path = TaskDataset.ROOT_DIR / ".temp"
        try:
            os.makedirs(temp_path, exist_ok=True)
            if num_workers <= 1:
                print(
                    f"----->>> {'Start' if len(downloaded_subject_ids) == 0 else 'Continue'} downloading dataset. <<<-----"
                )
                TaskDataset._download_dataset(
                    subject_ids=required_subject_ids,
                    secrete=secrete,
                    progress=shared_progress,
                    excluded_data=excluded_data,
                )
                TaskDataset._print_progress(progress=shared_progress)
                print()
            else:
                print(
                    f"----->>> {'Start' if len(downloaded_subject_ids) == 0 else 'Continue'} downloading dataset. workers: {num_workers} <<<-----"
                )
                aws.set_concurrent_requests(max_requests=num_workers)

                random.shuffle(required_subject_ids)
                chunk_size = len(required_subject_ids) // num_workers
                subject_ids_subsets = [
                    required_subject_ids[i * chunk_size : (i + 1) * chunk_size]
                    for i in range(num_workers)
                ]

                if len(required_subject_ids) % num_workers:
                    subject_ids_subsets[-1].extend(
                        required_subject_ids[num_workers * chunk_size :]
                    )

                args_list = [
                    (subset, secrete, shared_progress, excluded_data)
                    for subset in subject_ids_subsets
                ]

                with Pool(processes=num_workers) as pool:
                    pool.map(TaskDataset._download_subset, args_list)
                    print()
        finally:
            os.rmdir(path=temp_path)

        subject_data_paths = get_folders(TaskDataset.ROOT_DIR)
        n_subjects = len(subject_data_paths)
        n_tasks = [0] * len(TaskDataset.TASKS)
        for subject_data_path in subject_data_paths:
            for i, task_path in enumerate(get_folders(subject_data_path)):
                n_tasks[i] += len(get_files(task_path))

        meta_data = {"n_subjects": n_subjects, "n_tasks": sum(n_tasks)}
        meta_data.update({f"n_{task}": 0 for task in TaskDataset.TASKS})
        for task, task_n in zip(TaskDataset.TASKS, n_tasks):
            meta_data[f"n_{task}"] += task_n
        excluded_data = dict(excluded_data)
        excluded_data["path"] = sorted(excluded_data["path"])
        excluded_data = dict(
            sorted(excluded_data.items(), key=lambda item: (item[0] == "path", item[0]))
        )
        meta_data["excluded_data"] = dict(excluded_data)
        json.write(path=TaskDataset.META_PATH, data=meta_data)

        print("----->>> Finished downloading dataset. <<<-----")

    @staticmethod
    def _print_progress(progress):
        sys.stdout.write(
            f"\rdownloaded subjects: {progress['downloaded']}/~{progress['total']}"
        )
        sys.stdout.flush()

    @staticmethod
    def _download_dataset(subject_ids, secrete, progress, excluded_data):
        client = aws.get_client(secrete=secrete)
        temp_path = TaskDataset.ROOT_DIR / ".temp"

        for i, subject_id in enumerate(subject_ids):
            subject_data_path = TaskDataset.ROOT_DIR / subject_id
            temp_mri_file_path = temp_path / f"temp_mri_{subject_id}.nii.gz"
            try:
                for j, (task, ev) in enumerate(
                    zip(TaskDataset.TASKS, TaskDataset.TASK_EVS)
                ):
                    mri_file_path = TaskDataset.MRI_FILE_FORMAT.format(
                        id=subject_id, task=task
                    )
                    ev_file_path = TaskDataset.EV_FILE_FORMAT.format(
                        id=subject_id, task=task, ev=ev
                    )
                    try:
                        aws.download(
                            client=client,
                            bucket=TaskDataset.BUCKET_NAME,
                            file_path=mri_file_path,
                            local_file_path=temp_mri_file_path,
                        )
                    except FileNotFoundError:
                        continue
                    mri_file = nib.load(filename=temp_mri_file_path)
                    mri = mri_file.get_fdata(dtype=np.float32)
                    mri_cropped = mri[
                        TaskDataset.BRAIN_BOUNDARIES[
                            "x_start"
                        ] : TaskDataset.BRAIN_BOUNDARIES["x_end"],
                        TaskDataset.BRAIN_BOUNDARIES[
                            "y_start"
                        ] : TaskDataset.BRAIN_BOUNDARIES["y_end"],
                        TaskDataset.BRAIN_BOUNDARIES[
                            "z_start"
                        ] : TaskDataset.BRAIN_BOUNDARIES["z_end"],
                    ]
                    try:
                        ev_file = aws.get_file(
                            client=client,
                            bucket=TaskDataset.BUCKET_NAME,
                            file_path=ev_file_path,
                        ).decode("utf-8")
                    except FileNotFoundError:
                        os.remove(temp_mri_file_path)
                        continue

                    lines = ev_file.strip().split("\n")
                    task_meta = [list(map(float, line.split())) for line in lines]

                    subject_task_path = subject_data_path / task
                    os.makedirs(name=str(subject_task_path))

                    for k, (onset, duration, _) in enumerate(task_meta):
                        k += 1

                        if k > TaskDataset.TASK_LIMIT:
                            break

                        start = math.floor(onset / TaskDataset.TR)
                        end = math.ceil(
                            (onset + duration + TaskDataset.DELTA) / TaskDataset.TR
                        )
                        mri_cropped_ = mri_cropped[..., start : (end + 1)]
                        file_name = f"{subject_id}_{task}_{k}.npy"

                        if mri_cropped_.shape[-1] < 27:
                            key = f"{task}_{k}"
                            if key not in excluded_data:
                                excluded_data[key] = 1
                            else:
                                excluded_data[key] += 1
                            excluded_data["path"].append(file_name)
                            break

                        task_file_path = subject_task_path / file_name
                        torch.save(
                            obj=torch.tensor(mri_cropped_, dtype=torch.float32).permute(
                                3, 0, 1, 2
                            ),
                            f=task_file_path,
                        )

                    os.remove(path=temp_mri_file_path)
            except BaseException:
                if subject_data_path.exists():
                    shutil.rmtree(path=subject_data_path)
                if temp_mri_file_path.exists():
                    os.remove(path=temp_mri_file_path)
                raise

            if subject_data_path.exists():
                progress["downloaded"] += 1
            else:
                progress["total"] -= 1
            TaskDataset._print_progress(progress=progress)

    @staticmethod
    def _download_subset(args):
        subject_ids_, secrete_, shared_progress, excluded_data = args
        TaskDataset._print_progress(progress=shared_progress)
        TaskDataset._download_dataset(
            subject_ids=subject_ids_,
            secrete=secrete_,
            progress=shared_progress,
            excluded_data=excluded_data,
        )

    def _load_samples(self):
        samples = []
        for subject_id in self.subject_ids:
            subject_path = self.ROOT_DIR / subject_id
            for task_folder in subject_path.iterdir():
                for file in task_folder.glob("*.npy"):
                    samples.append((str(file), task_folder.name))
        return samples


if __name__ == "__main__":
    TaskDataset.download_dataset(num_workers=max(1, get_num_physical_cores() - 1))
