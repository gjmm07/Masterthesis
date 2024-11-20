from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import utils
import os
from ViconMoCap import get_joint
from itertools import cycle, chain
import warnings

def read_mocap_marker(base_path: os.PathLike or str):
    path = os.path.join(base_path, "Mocap")
    return utils.read_data(os.path.join(path, "marker.h5"))


def read_mocap_joints(base_path: os.PathLike or str):
    path = os.path.join(base_path, "Mocap")
    return utils.read_data(os.path.join(path, "joint_data.h5"))


def read_ort_data(base_path: os.PathLike or str):
    return dict(reversed(list(utils.read_data(os.path.join(base_path, "Ort_data", "data.h5")).items())))


def read_emg_data(base_path: os.PathLike or str):
    path = os.path.join(base_path, "EMG_data")
    data = {}
    for dir_ in os.listdir(path):
        subdir = os.path.join(path, dir_)
        if not os.path.isdir(subdir):
            continue
        data[dir_] = utils.read_data(os.path.join(subdir, "data.h5"))
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
                except KeyError:
                    pass
    return info

def plot_joints(yname, joint_data, ax: plt.Axes, time: int or None):
    for name, data in joint_data.items():
        if time is None:
            time = 1
        ax.plot(np.linspace(
                0, time, data.shape[0]),
                data, label=name)
    ax.legend()
    ax.set_ylabel(yname)


def plot_emg(sensor: str, chan_data, ax: plt.Axes, time: int or None):
    for offset, (chan, data) in enumerate(chan_data.items()):
        if time is None:
            # todo: fix this
            time = 1
        print(np.mean(data))
        ax.plot(np.linspace(0, time, data.shape[0]),
                data + (offset / (len(chan_data)- 1)), lw=0.2, label=chan)
        ax.set_ylabel(f"{sensor} [mV]")
        ax.legend()


def get_joints(path: os.PathLike or str):
    return (("MoCap Joints [deg]", ),
            (dict(zip(("upper arm", "lower arm"), read_mocap_joints(path)["joints"].T)), ))


def get_ort_joints(path: os.PathLike or str):
    data = read_ort_data(path)
    data.pop("load_cell")
    return ("Ort Joints [deg]", ), (data, )


def get_emg_data(path: os.PathLike or str):
    data = []
    read_data = read_emg_data(path)
    for sensor_name, sensor_data in read_data.items():
        emg_chan_data = {}
        for chan_name, chan_data in sensor_data.items():
            if "emg" in chan_name.lower():
                print(chan_name)
                emg_chan_data[chan_name] = chan_data
        data.append(emg_chan_data)
    return tuple(read_data.keys()), data


def plot_dataset(path, to_plot: tuple[bool, bool, bool]):
    if not to_plot[0]:
        warnings.warn("If Mocap data is not plotted, y axis will be set to samples")
    to_plot = cycle(to_plot)
    funcs = (get_joints, get_ort_joints, get_emg_data)
    plot_names, data = [], []
    for func, tp in zip(funcs, to_plot):
        if tp:
            plot_name, read_data = func(path)
            plot_names += plot_name
            data += read_data
    fig, axs = plt.subplots(len(plot_names), 1)
    info = iter(zip(axs.flatten(), plot_names, data))
    time = None
    if next(to_plot):
        # joint_data = next(data)
        ax, name, joint_data = next(info)
        time = time = joint_data["upper arm"].shape[0] / 100
        plot_joints(name, joint_data, ax, time)
    if next(to_plot):
        # ort_data = next(data)
        ax, name, ort_data = next(info)
        plot_joints(name, ort_data, ax, time)
    if next(to_plot):
        for ax, name, emg_data in info:
            plot_emg(name, emg_data, ax, time)
    plt.show()


def plot_markers(path: os.PathLike or str):
    marker_names = [x[0] for x in read_mocap_setup(path)["Markers"]]
    print(marker_names)
    data = read_mocap_marker(path)
    markers = data["mocap"]
    predicted = data["marker_prediction"]
    markers = markers - markers[0, 0, :]
    markers = markers[10:20]
    print(np.all(predicted[10:20] == 1))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i, name in enumerate(marker_names):
        ax.text(markers[0, i, 0], markers[0, i, 1], markers[0, i, 2], s=name, size=5, zorder=1)
    ax.plot(*markers[0, [0, 1], :].T)
    ax.plot(*markers[0, [2, 4], :].T)
    ax.plot(*markers[0, [6, 5], :].T)
    center_shoulder, _ = get_joint(markers[0, 1, :], markers[0, 0, :])
    center_elbow, _ = get_joint(markers[0, 2, :], markers[0, 4, :])
    center_wrist, _ = get_joint(markers[0, 5, :], markers[0, 6, :])
    centers = np.array((center_shoulder, center_elbow, center_wrist))
    # ax.scatter(*centers.T)
    ax.plot(*centers.T, marker="o")
    for i in range(markers.shape[1]):
        ax.plot(*markers[:, i, :].T, markersize=1, marker="o")
    plt.show()


if __name__ == "__main__":
    # print(get_emg_data("recordings/18-11-24--17-42-35"))
    plot_dataset("recordings/20-11-24--16-44-53", (False, False, True))
    # plot_markers("recordings/18-11-24--17-42-35")