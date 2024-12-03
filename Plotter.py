from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import utils
import os
from ViconMoCap import get_joint, calc_elbow_angle, calc_wrist_angle
from itertools import cycle, chain
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings

EXPORT_TIME = (60, 120)


def read_mocap_marker(base_path: os.PathLike or str):
    path = os.path.join(base_path, "MoCap")
    return utils.read_data(os.path.join(path, "marker.h5"))


def read_mocap_joints(base_path: os.PathLike or str):
    path = os.path.join(base_path, "MoCap")
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
                except (KeyError, SyntaxError):
                    pass
    return info

def plot_joints(yname, joint_data, ax: plt.Axes, time: int or None):
    plot_data = None
    for name, data in joint_data.items():
        if time is None:
            time = 1
        plot_data = np.stack((np.linspace(0, time, data.shape[0]), data.flatten()), axis=1)
        ax.plot(plot_data[:, 0], plot_data[:, 1], label=name)
    ax.legend()
    ax.set_ylabel(yname)
    return plot_data

def plot_emg(sensor: str, chan_data, ax: plt.Axes, time: int or None):
    if time is None:
        # todo: fix this
        time = 1
    plot_data = []
    for offset, data in enumerate(chan_data.values()):
        if not offset:
            plot_data.append(np.atleast_2d(np.linspace(0, time, data.shape[0])).T)
        plot_data.append(data + (offset / (len(chan_data)- 1)))
    plot_data = np.hstack(plot_data)
    ax.plot(plot_data[:, 0], plot_data[:, 1:], lw=0.2, label=chan_data.keys())
    ax.set_ylabel(f"{sensor} [mV]")
    ax.legend()
    return plot_data

def plot_markers(ax: plt.Axes, marker_label, data, time):
    ax.set_ylabel(f"{marker_label} [mm]")
    marker_pos = data["mocap"]
    plot_data = np.hstack((
        np.atleast_2d(np.linspace(0, time, marker_pos.shape[0])).T,
        marker_pos
    ))
    ax.plot(plot_data[:, 0],
            plot_data[:, 1:], label=("x", "y", "z"))
    ax.legend()
    return plot_data


def get_joints(path: os.PathLike or str) -> tuple[tuple[str], tuple[dict], float]:
    data = read_mocap_joints(path)["joints"].T
    time = data.shape[1] / 100
    return (("MoCap Joints [deg]", ),
            (dict(zip(("upper arm", "lower arm"), data)), ), time)


def get_markers(path: os.PathLike or str, *marker_labels):
    all_marker_labels = [x[0] for x in read_mocap_setup(path)["Markers"]]
    # 'ShoulderB', 'ShoulderF', 'ElbowO', 'UpperArm', 'ElbowI', 'WristI', 'WristO', 'ForeArm', 'WristL'
    if marker_labels == ():
        marker_labels = all_marker_labels
    idx = [all_marker_labels.index(x) for x in marker_labels]
    data = []
    raw_data = read_mocap_marker(path)
    mocap = raw_data["mocap"]
    mocap = mocap - mocap[:, 0:1, :]
    time = raw_data["marker_prediction"].shape[0] / 100
    for i in idx:
        data.append({"marker_prediction": raw_data["marker_prediction"][:, i],
                     "mocap": mocap[:, i, :]})
    return marker_labels, data, time


def get_ort_joints(path: os.PathLike or str):
    data = read_ort_data(path)
    data.pop("load_cell")
    return ("Ort Joints [deg]", ), (data, ), None


def get_emg_data(path: os.PathLike or str):
    data = []
    read_data = read_emg_data(path)
    for sensor_name, sensor_data in read_data.items():
        emg_chan_data = {}
        for chan_name, chan_data in sensor_data.items():
            if "emg" in chan_name.lower():
                emg_chan_data[chan_name] = chan_data
        data.append(emg_chan_data)
    return tuple(read_data.keys()), data, None

def _export_txt(filename: str, array: np.ndarray, *args, **kwargs):
    array = array[(EXPORT_TIME[0] < array[:, 0]) & (array[:, 0] <= EXPORT_TIME[1])]
    array[:, 0] -= array[0, 0]
    np.savetxt(filename, array, *args, **kwargs)


def plot_dataset(path, to_plot: tuple[bool, bool, bool, bool], export_csvs: bool = False):
    if not to_plot[0]:
        warnings.warn("If Mocap data is not plotted, y axis will be set to samples")
    to_plot = cycle(to_plot)
    funcs = (get_joints,
             get_ort_joints,
             lambda p: get_markers(p,
                                   "ElbowO", "WristI"
                                   ),
             get_emg_data)
    plot_names, data, ns = [], [], []
    time = None
    for func, tp in zip(funcs, to_plot):
        if tp:
            plot_name, read_data, t = func(path)
            if t is not None:
                time = t
            ns.append(len(read_data))
            plot_names += plot_name
            data += read_data
    print(time)
    fig, axs = plt.subplots(len(plot_names), 1, squeeze=False)
    info = iter(zip(axs.flatten(), plot_names, data))
    ns = iter(ns)
    if next(to_plot):
        # joint_data = next(data)
        _ = next(ns)
        ax, name, joint_data = next(info)
        p_data = plot_joints(name, joint_data, ax, time)
        if export_csvs:
            _export_txt(name, p_data, header="t, y")
    if next(to_plot):
        _ = next(ns)
        ax, name, ort_data = next(info)
        p_data = plot_joints(name, ort_data, ax, time)
        if export_csvs:
            _export_txt(name, p_data, header="t, y")
    if next(to_plot):
        for _ in range(next(ns)):
            ax, name, marker_data = next(info)
            p_data = plot_markers(ax, name, marker_data, time)
            if export_csvs:
                _export_txt(name, p_data, header="t, x, y, z")
    if next(to_plot):
        for _ in range(next(ns)):
            ax, name, emg_data = next(info)
            p_data = plot_emg(name, emg_data, ax, time)
            if export_csvs:
                _export_txt(name, p_data, header="t, a, b, c, d")
    plt.show()

def _plot_elbow_angle(ax: plt.Axes, c_shoulder, c_elbow, c_wrist):
    radius = 40
    center = c_elbow
    n = 100
    angle = 180 - calc_elbow_angle(c_shoulder, c_elbow, c_wrist)
    v1 = c_wrist - center
    v2 = c_shoulder - center
    x = np.atleast_2d(np.linspace(0, np.radians(angle), n))
    n_plane = np.cross(v1, v2)
    n2 = np.cross(n_plane, v1)
    v1 = v1 / np.linalg.norm(v1)
    n2 = n2 / np.linalg.norm(n2)
    arc_elbow = center + radius * np.sin(x.T) * n2 + radius * np.cos(x.T) * v1
    ax.plot(*arc_elbow.T)
    vertices = np.stack(
        (arc_elbow[1:], arc_elbow[:-1], np.tile(c_elbow, (n-1, 1))), axis=2).transpose(0, 2, 1)
    vertices = Poly3DCollection(vertices)
    ax.add_collection3d(vertices)


def _plot_wrist_angle(ax: plt.Axes, c_elbow, c_wrist, elbow_axis, wrist_axis):
    radius = 30
    n = 100
    angle = 180 - calc_wrist_angle(c_elbow, c_wrist, elbow_axis, wrist_axis)
    x = np.atleast_2d(np.linspace(0, np.radians(angle), n))
    n_plane = (c_wrist - c_elbow) / np.linalg.norm(c_elbow - c_wrist)
    n2 = np.cross(n_plane, wrist_axis)
    n2 = n2 / np.linalg.norm(n2)
    arc_wrist = c_wrist + radius * np.sin(x.T) * n2 + radius * np.cos(x.T) * wrist_axis / np.linalg.norm(wrist_axis)
    ax.plot(*arc_wrist.T)
    vertices = np.stack(
        (arc_wrist[1:], arc_wrist[:-1], np.tile(c_wrist, (n - 1, 1))), axis=2).transpose(0, 2, 1)
    vertices = Poly3DCollection(vertices)
    ax.add_collection3d(vertices)


def _get_axis_limits(markers: np.ndarray):
    fig_center = (np.max(markers, axis=(0, 1)) + np.min(markers, axis=(0, 1))) / 2
    axis_len = max(np.max(markers, axis=(0, 1)) - np.min(markers, axis=(0, 1)))
    axis_limits = np.atleast_2d(fig_center).T + np.atleast_2d([-axis_len / 2 - 10, axis_len / 2 + 10])
    return axis_limits


def _output_latex(ary: np.ndarray):
    if ary.ndim == 2:
        print("{", end="")
        for row in ary:
            print("(", end="")
            for i, val in enumerate(row):
                print(val, end="")
                if i != 2:
                    print(", ", end="")
            print(") ")
        print("}")

def _output_arc_fill_latex(center: np.ndarray, ary: np.ndarray):
    ary = np.vstack((center, ary))
    for row in ary:
        print("(", end="")
        for i, val in enumerate(row):
            print("%.3f" % val, end="")
            if i != 2:
                print(", ", end="")
        print(") -- ")


def _output_tikz(fig):
    import tikzplotlib
    tikzplotlib.save("test.tex")


def _read_markers(path: os.PathLike or str):
    marker_names = [x[0] for x in read_mocap_setup(path)["Markers"]]
    data = read_mocap_marker(path)
    markers = data["mocap"]
    predicted = data["marker_prediction"]
    return marker_names, markers, predicted

def plot_markers3d(path: os.PathLike or str):
    marker_names, markers, predicted = _read_markers(path)
    markers = markers - markers[0, 0, :]
    markers = markers[10:20]
    print(np.all(predicted[10:20] == 1))
    fig = plt.figure()
    axis_limits = _get_axis_limits(markers)
    ax = fig.add_subplot(projection="3d")
    ax.axes.set_xlim3d(axis_limits[0, :])
    ax.axes.set_ylim3d(axis_limits[1, :])
    ax.axes.set_zlim3d(axis_limits[2, :])
    ax.set_aspect("equal")
    for i, name in enumerate(marker_names):
        ax.text(markers[0, i, 0], markers[0, i, 1], markers[0, i, 2], s=name, size=10, zorder=1)
    ax.plot(*markers[0, [0, 1], :].T)
    ax.plot(*markers[0, [2, 4], :].T)
    ax.plot(*markers[0, [6, 5], :].T)
    center_shoulder, _ = get_joint(markers[0, 1, :], markers[0, 0, :])
    center_elbow, elbow_axis = get_joint(markers[0, 2, :], markers[0, 4, :])
    center_wrist, wrist_axis = get_joint(markers[0, 5, :], markers[0, 6, :])
    centers = np.array((center_shoulder, center_elbow, center_wrist))
    ax.plot(*centers.T, marker="o")
    _plot_elbow_angle(ax, center_shoulder, center_elbow, center_wrist)
    _plot_wrist_angle(ax, center_elbow, center_wrist, elbow_axis, wrist_axis)
    for i in range(markers.shape[1]):
        ax.plot(*markers[:, i, :].T, c="blue", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # print(get_emg_data("recordings/18-11-24--17-42-35"))
    plot_dataset("recordings/03-12-24--18-46-14", (True, False, False, True))
    # plot_markers("recordings/20-11-24--16-26-08")