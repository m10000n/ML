from data.HCP_1200.hcp_1200_data import HCP1200Data
from model.Wang.architecture.original import OriginalConfig
from model.Wang.architecture.original2 import Original2Config

_ORIGINAL_N_VOLUMES = 27


# original
def original_prep() -> OriginalConfig:  # 2,598,049 params
    return OriginalConfig(
        description="prepcoressed fMRIs",
        input_shape=(_ORIGINAL_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
    )


def original_unp() -> OriginalConfig:  # 2,843,809 params
    return OriginalConfig(
        description="unprocessed fMRIs",
        input_shape=(_ORIGINAL_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
    )


# original2
def original2_prep() -> Original2Config:  # 3,982,177 params
    return Original2Config(
        description="prepcoressed fMRIs",
        input_shape=(_ORIGINAL_N_VOLUMES, *HCP1200Data.SHAPE_PREPROCESSED),
        drop_out=False,
    )


def original2_unp() -> Original2Config:  # 4,227,937 params
    return Original2Config(
        description="unprocessed fMRIs",
        input_shape=(_ORIGINAL_N_VOLUMES, *HCP1200Data.SHAPE_UNPROCESSED),
        drop_out=False,
    )
