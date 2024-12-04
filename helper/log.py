from datetime import datetime

from helper import json, system


class Log:
    def __init__(self, file_path):
        self.file_path = file_path
        self.start_time = datetime.now()
        self.end_time = self.start_time
        self.system = {
            "cpu": system.get_cpu_info(),
            "gpu": system.get_gpu_info(),
        }
        self.data = {"ids": {"train": [], "val": [], "test": []}}
        self.n_epochs = 0
        self.best_model_epoch = 0
        self.learning_rate = []
        self.loss = {"train": [], "val": [], "test": -1}
        self.accuracy = {"val": [], "test": -1}
        self.confusion = {"predicted": [], "actual": []}
        self.debug = {}
        self.update()

    def set_dataset_ids(self, ids):
        self.data["ids"] = ids

    def set_loss(self, train=None, val=None, test=None):
        if train:
            self.loss["train"] = train
        if val:
            self.loss["val"] = val
        if test:
            self.loss["test"] = test

    def set_accuracy(self, val=None, test=None):
        if val:
            self.accuracy["val"] = val
        if test:
            self.accuracy["test"] = test

    def set_confusion(self, predicted, actual):
        self.confusion["predicted"] = predicted
        self.confusion["actual"] = actual

    def update(self):
        data = {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "system": self.system,
            "n_epochs": self.n_epochs,
            "best_model_epoch": self.best_model_epoch,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "confusion": self.confusion,
            "data": self.data,
            "debug": self.debug,
        }
        json.write(file_path=self.file_path, data=data)

    def end(self):
        self.end_time = datetime.now()
        self.update()
