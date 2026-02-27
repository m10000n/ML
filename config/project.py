# this file is used before the python environment is ready

from typing import Union

##### config start #####
_MODEL: Union[str, None] = "Inceptron"

_DATASET: Union[str, None] = "HCP_1200"

_PROVIDER: Union[str, None] = None
##### config end #####


def get_model() -> str:
    if not _MODEL:
        raise ValueError("Model not specified.")
    return _MODEL


def get_dataset() -> str:
    if not _DATASET:
        raise ValueError("Dataset not specified.")
    return _DATASET


def get_provider() -> str:
    if not _PROVIDER:
        raise ValueError("Provider not specified.")
    return _PROVIDER
