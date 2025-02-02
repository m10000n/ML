from torch.utils.data import Dataset

from helper.aws import download
from helper.path import DATASET_PATH_A
from datasets import load_dataset

class OpusDataset(Dataset):
    DATASET_NAME = "Helsinki-NLP/opus_books"
    DATA_PATH = DATASET_PATH_A / "data"

    def __init__(self, subset: str):
        self.subsets = subset

        self._download_dataset(subset)

    def _download_dataset(self, subset: str):
        load_dataset(path=self.DATASET_NAME, name=subset, split="train", cache_dir=self.DATA_PATH / f"{subset}.json")




if __name__ == "__main__":
    d = OpusDataset("de-en")
