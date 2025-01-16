import os
import shutil
import stat


def get_pu(world_size, rank):
    if world_size == 0:
        return "cpu"
    elif world_size == 1:
        return "cuda"
    else:
        return rank


def save_exp_files(exp_file_folder, files):
    for source, file_name in files:
        destination = exp_file_folder / file_name
        shutil.copy(src=source, dst=exp_file_folder / file_name)
        os.chmod(str(destination), stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    shutil.make_archive(
        base_name=exp_file_folder, format="zip", root_dir=exp_file_folder
    )
    shutil.rmtree(exp_file_folder)
