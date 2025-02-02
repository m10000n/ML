from torch.cuda.tunable import write_file_on_exit
from torch.utils.data import Dataset
from kagglehub import dataset_download
from helper.path import dir_path, exists
from os import makedirs
from shutil import rmtree
from pathlib import Path
from helper.print import print_start, print_end
from helper.path import get_files
from pandas import read_csv, concat
from chardet import detect
from pandas import read_csv
from helper.file import combine_files



class WMTDataset(Dataset):
    DATASET_URL = "mohamedlotfy50/wmt-2014-english-german"
    DATASET_DIR = dir_path() / "data"
    DATASET_PATH = DATASET_DIR / "e-g.csv"

    def __init__(self):
        return

    @staticmethod
    def download_dataset():
        if not exists(WMTDataset.DATASET_PATH):
            print_start("Start downloading dataset.")
            makedirs(WMTDataset.DATASET_DIR, exist_ok=True)
            kaggle_dataset_path = Path(dataset_download(WMTDataset.DATASET_URL))
            combine_files(source=get_files(kaggle_dataset_path), destination=WMTDataset.DATASET_PATH)
            rmtree(kaggle_dataset_path.parent.parent.parent)
            print_end("Finished downloading dataset.")







if __name__=="__main__":
    WMTDataset.download_dataset()