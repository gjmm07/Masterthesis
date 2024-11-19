import matplotlib.pyplot as plt
import numpy as np
import utils
import os
from ViconMoCap import get_joint


def read_mocap(base_path: os.PathLike or str):
    path = os.path.join(base_path, "Mocap")
    joint_data = utils.read_data(os.path.join(path, "joint_data.h5"))
    marker_data = utils.read_data(os.path.join(path, "marker.h5"))
    return joint_data, marker_data


def read_ort_data(base_path: os.PathLike or str):
    return utils.read_data(os.path.join(base_path, "Ort_data", "data.h5"))


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

def plot_dataset(path):
    joint_data, _ = read_mocap(path)
    ort_data = read_ort_data(path)
    emg_data = read_emg_data(path)
    fig, axs = plt.subplots(2 + len(emg_data), 1, sharex=True)
    time = joint_data["joints"].shape[0] / 100
    plt.xlabel("Time")
    axs[0].plot(np.tile(
        np.linspace(
            0, time, joint_data["joints"].shape[0]), (2, 1)).T,
        joint_data["joints"], label=("upper arm", "lower arm"))
    axs[0].legend()
    axs[0].set_ylabel("MoCap Angle [deg]")
    axs[1].plot(np.linspace(0, time, ort_data["angle_upper_arm"].shape[0]),
                ort_data["angle_upper_arm"], label="upper arm")
    axs[1].plot(np.linspace(0, time, ort_data["angle_lower_arm"].shape[0]),
                ort_data["angle_lower_arm"], label="lower arm")
    axs[1].legend()
    axs[1].set_ylabel("Ort Angle [deg]")
    for i, (sensor, chan_data) in enumerate(emg_data.items()):
        for offset, (chan, data) in enumerate(chan_data.items()):
            if "emg" in chan.lower():
                axs[2 + i].plot(np.linspace(0, time, data.shape[0]),
                                data + offset, lw=0.2, label=chan)
                axs[2 + i].set_ylabel(f"{sensor} [mV]")
                axs[2 + i].legend()
    plt.show()


def plot_markers(path: os.PathLike or str):
    marker_names = [x[0] for x in read_mocap_setup(path)["Markers"]]
    print(marker_names)
    _, data = read_mocap(path)
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
    plot_dataset("recordings/18-11-24--17-42-35")
    # plot_markers("recordings/18-11-24--17-42-35")