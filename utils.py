import h5py
import os
import numpy as np
from typing import Sequence


def save(path: os.PathLike or str,
         data: np.ndarray or Sequence[np.ndarray],
         dset_name: str or Sequence[str]):
    if isinstance(data, np.ndarray):
        data = [data]
    if isinstance(dset_name, str):
        dset_name = [dset_name]
    with h5py.File(path, "a") as file:
        for name, d in zip(dset_name, data):
            dset = file[name]
            dset.resize((dset.shape[0] + d.shape[0]), axis=0)
            dset[-d.shape[0]:, :] = d


def read_data(path: os.PathLike or str):
    with h5py.File(path, "r") as file:
        keys = list(file.keys())
        values = [np.array(file[key]) for key in keys]
    return dict(zip(keys, values))


def read_mocap_marker(base_path: os.PathLike or str):
    path = os.path.join(base_path, "MoCap")
    return read_data(os.path.join(path, "marker.h5"))


def read_mocap_joints(base_path: os.PathLike or str):
    path = os.path.join(base_path, "MoCap")
    return read_data(os.path.join(path, "joint_data.h5"))


def read_ort_data(base_path: os.PathLike or str):
    return dict(reversed(list(read_data(os.path.join(base_path, "Ort_data", "data.h5")).items())))


def read_emg_data(base_path: os.PathLike or str):
    path = os.path.join(base_path, "EMG_data")
    data = {}
    for dir_ in os.listdir(path):
        subdir = os.path.join(path, dir_)
        if not os.path.isdir(subdir):
            continue
        data[dir_] = read_data(os.path.join(subdir, "data.h5"))
    return data


def read_mocap_setup(base_path: os.PathLike or str):
    info = {}
    with open(os.path.join(base_path, "MoCap", "setup.txt"), "r") as file:
        key = ""
        for line in file.readlines():
            if line.startswith("###"):
                key = str(line.strip("# \n"))
                info[key] = []
            else:
                try:
                    info[key].append(eval(line))
                except NameError:
                    info[key].append(line.strip())
                except (KeyError, SyntaxError):
                    pass
    return info

