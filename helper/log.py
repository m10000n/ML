import os
import stat
from datetime import datetime

from helper import system


class Log:
    def __init__(self, file_path, save_on_create=True):
        self.file_path = file_path
        self.start_time = datetime.now()
        self.end_time = self.start_time
        self.system = {
            "cpu": system.get_cpu_info(),
            "gpu": system.get_gpu_info(),
        }
        self.data = {"ids": {"train": [], "val": [], "test": []}}
        self.n_epochs = 0
        self.learning_rate = []
        self.loss = {"train": [], "val": [], "test": -1}
        self.accuracy = {"val": [], "test": -1}
        self.confusion = {"predicted": [], "actual": []}
        self.debug = {}

        if save_on_create:
            self.update()

    def __str__(self):
        return str(self.as_dict())

    @staticmethod
    def from_json(path):
        log_json = json.read(path=path)
        log = Log(file_path=path, save_on_create=False)
        log.start_time = datetime.fromisoformat(log_json["start_time"])
        log.end_time = datetime.fromisoformat(log_json["end_time"])
        log.system = log_json["system"]
        log.data = log_json["data"]
        log.n_epochs = log_json["n_epochs"]
        log.learning_rate = log_json["learning_rate"]
        log.loss = log_json["loss"]
        log.accuracy = log_json["accuracy"]
        log.confusion = log_json["confusion"]
        log.debug = log_json["debug"]
        return log

    def get_loss_val(self):
        return self.loss["val"]

    def get_accuracy_val(self):
        return self.accuracy["val"]

    def get_accuracy_test(self):
        return self.accuracy["test"]

    def get_confusion_actual(self):
        return self.confusion["actual"]

    def get_confusion_predicted(self):
        return self.confusion["predicted"]

    def set_data_ids(self, ids):
        self.data["ids"] = ids

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

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

    def as_dict(self):
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "system": self.system,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "confusion": self.confusion,
            "data": self.data,
            "debug": self.debug,
        }

    def update(self):
        json.write(path=self.file_path, data=self.as_dict())

    def end(self):
        self.end_time = datetime.now()
        self.update()
        os.chmod(str(self.file_path), stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
