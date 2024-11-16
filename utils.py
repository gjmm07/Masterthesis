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
    return keys, values

