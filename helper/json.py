import json


def write(path, data):
    with open(path, "w") as file:
        json.dump(obj=data, fp=file, indent=4)


def read(path):
    with open(path, "r") as file:
        return json.load(fp=file)
