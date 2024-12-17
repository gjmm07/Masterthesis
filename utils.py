import h5py
import os
import numpy as np
from typing import Sequence
from operator import itemgetter
from dataclasses import dataclass


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
    if not os.path.exists(path):
        return
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


def read_subject(base_path: os.PathLike or str):
    with open(os.path.join(base_path, "info.txt"), "r") as f:
        return f.readline().strip()


def read_dataset(
        subject: str,
        *,
        timestamp: str or None = None,
        read_mocap: bool = True,
        read_emg: bool = True,
        read_ort: bool = False
):
    # todo: Handle cases where not data is present
    emg_data, mocap, marker_pred, mocap_joints = [], [], [], []
    upper_mot_angle, lower_mot_angle, load_cell = [], [], []
    marker_labels = None
    for dir_ in os.listdir("recordings"):
        if timestamp is not None and dir_[:len(timestamp)] != timestamp:
            continue
        path = os.path.join("recordings", dir_)
        if read_subject(path) != subject:
            continue
        if read_mocap:
            marker_prediction, m_mocap = itemgetter(
                "marker_prediction", "mocap")(read_mocap_marker(path))
            mocap_joints.append(read_mocap_joints(path)["joints"])
            marker_pred.append(marker_prediction)
            mocap.append(m_mocap)
            if marker_labels is None:
                marker_labels = [x[1] for x in read_mocap_setup(path)["Markers"]]
        if read_emg:
            emg_data.append(
                np.array([x.flatten() for y in list(read_emg_data(path).values()) for x in y.values()]).T)
        if read_ort:
            um_angle, lm_angle, lc = itemgetter("angle_upper_arm", "angle_lower_arm", "load_cell")(read_ort_data(path))
            upper_mot_angle.append(um_angle)
            lower_mot_angle.append(lm_angle)
            load_cell.append(lc)
    return Data(mocap_joints, emg_data, marker_pred, mocap, marker_labels, upper_mot_angle, lower_mot_angle, load_cell)


@dataclass
class Data:
    mocap_joints: list
    emg_data: list
    marker_prediction: list
    mocap: list
    marker_labels: list
    upper_motor_angle: list
    lower_motor_angle: list
    load_cell: list

    imp_marker_idx = (0, 1, 2, 4, 5, 6) # important markers

    @staticmethod
    def _crop_data(data: np.ndarray, bounds: np.ndarray):
        bounds = (bounds * data.shape[0]).astype(int)
        return data[bounds[0]:bounds[1]]

    def drop_useless(self):
        mocap_joints, mocap, marker_prediction = [], [], []
        emg_data, upper_motor_angle, lower_motor_angle, load_cell = [], [], [], []
        for m_joints, marker, marker_pred, emg_d, um_angle, lm_angle, lc in (
                zip(
                    self.mocap_joints, self.mocap, self.marker_prediction, self.emg_data,
                    self.upper_motor_angle, self.lower_motor_angle, self.load_cell)):
            mask = (
                np.any(np.sqrt(
                    np.sum((marker[1:, :, :] - marker[:-1,:,:])**2, axis=2))[:, self.imp_marker_idx] > 50, axis=1) |
                np.any(marker_pred[:-1] == 0, axis=1) |
                np.any(np.diff(m_joints, axis=0) > 30, axis=1))
            mask = [0] + list(np.where(mask)[0]) + [m_joints.shape[0]]
            for i, diff in enumerate(np.diff(mask)):
                if diff > 1000:
                    bounds = (mask[i] + 1, mask[i + 1] - 1)
                    mocap_joints.append(m_joints[bounds[0]: bounds[1]])
                    mocap.append(marker[bounds[0]: bounds[1]])
                    marker_prediction.append(marker_pred[bounds[0]: bounds[1]])
                    # handle other sample rates
                    bounds = np.array(bounds) / m_joints.shape[0]
                    for l, raw_data in zip(
                            (emg_data, upper_motor_angle, lower_motor_angle, load_cell),
                            (emg_d, um_angle, lm_angle, lc)):
                        l.append(self._crop_data(raw_data, bounds))
        self.mocap_joints = mocap_joints
        self.mocap = mocap
        self.marker_prediction = marker_prediction
        self.emg_data = emg_data
        self.upper_motor_angle = upper_motor_angle
        self.lower_motor_angle = lower_motor_angle
        self.load_cell = load_cell


if __name__ == "__main__":
    d = read_dataset("Finn", timestamp="05-12-24", read_ort=False)
    d.drop_useless()
    import matplotlib.pyplot as plt
    for l, emg, lm, up in zip(d.mocap_joints, d.emg_data, d.lower_motor_angle, d.upper_motor_angle):
        fig, axs = plt.subplots(3)
        axs[0].plot(l)
        axs[1].plot(up)
        axs[1].plot(lm)
        axs[2].plot(emg + np.atleast_2d(np.linspace(0, 1, 8)))
        plt.show()


