import json


def write(file_path, data):
    with open(file_path, "w") as file:
        json.dump(obj=data, fp=file, indent=4)


def read(file_path):
    with open(file_path, "r") as file:
        return json.load(fp=file)
