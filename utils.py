import h5py
import os
import numpy as np


def save(path: os.PathLike or str, data: np.ndarray, dset_name: str):
    with h5py.File(path, "a") as file:
        dset = file[dset_name]
        dset.resize((dset.shape[0] + data.shape[0]), axis=0)
        dset[-data.shape[0]:, :] = data

def read_data(path: os.PathLike or str):
    with h5py.File(path, "r") as file:
        keys = list(file.keys())
        values = [np.array(file[key]) for key in keys]
    return keys, values

