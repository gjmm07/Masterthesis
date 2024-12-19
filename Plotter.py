from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import os
from ViconMoCap import get_joint, calc_elbow_angle, calc_wrist_angle
from itertools import cycle, chain
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings

from utils import read_mocap_joints, read_mocap_marker, read_ort_data, read_emg_data, read_mocap_setup

EXPORT_TIME = (193, 253)
_SAVE_PATH = "/home/finn/Documents/LatexMA/data/"
_EMG_PLOT_DIST: float = 0.3


def _export_txt(path: os.PathLike or str, array: np.ndarray,  *args, **kwargs):
    np.savetxt(path, array, *args, **kwargs, fmt="%f", delimiter=",", comments="")


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
    return arc_elbow


def _plot_wrist_angle(ax: plt.Axes, c_elbow, c_wrist, elbow_axis, wrist_axis):
    radius = 30
    n = 100
    angle = calc_wrist_angle(c_elbow, c_wrist, elbow_axis, wrist_axis)
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
    return arc_wrist


def _get_axis_limits(markers: np.ndarray):
    fig_center = (np.max(markers, axis=(0, 1)) + np.min(markers, axis=(0, 1))) / 2
    axis_len = max(np.max(markers, axis=(0, 1)) - np.min(markers, axis=(0, 1)))
    axis_limits = np.atleast_2d(fig_center).T + np.atleast_2d([-axis_len / 2 - 10, axis_len / 2 + 10])
    return axis_limits


def _output_arc_fill_latex(center: np.ndarray, ary: np.ndarray):
    ary = np.vstack((center, ary))
    for row in ary:
        print("(", end="")
        for i, val in enumerate(row):
            print("%.3f" % val, end="")
            if i != 2:
                print(", ", end="")
        print(") -- ")


def _read_markers(path: os.PathLike or str):
    marker_names = [x[0] for x in read_mocap_setup(path)["Markers"]]
    data = read_mocap_marker(path)
    markers = data["mocap"]
    predicted = data["marker_prediction"]
    return marker_names, markers, predicted


def plot_markers3d(path: os.PathLike or str,
                   t_point: int,
                   include_prev_samples: int = 10,
                   export_csvs: bool = False):
    marker_names, markers, predicted = _read_markers(path)
    markers -= markers[t_point, 0, :]
    markers = markers[t_point-include_prev_samples:t_point+1]
    if not np.all(predicted[t_point-include_prev_samples:t_point] == 1):
        warnings.warn("Some Markers are occluded or predicted")
    fig = plt.figure()
    axis_limits = _get_axis_limits(markers)
    ax = fig.add_subplot(projection="3d")
    ax.axes.set_xlim3d(axis_limits[0, :])
    ax.axes.set_ylim3d(axis_limits[1, :])
    ax.axes.set_zlim3d(axis_limits[2, :])
    ax.set_aspect("equal")
    for i, name in enumerate(marker_names):
        ax.text(markers[-1, i, 0], markers[-1, i, 1], markers[-1, i, 2], s=name, size=10, zorder=1)
    # Markers
    ax.scatter(*markers[-1,].T, marker="o")
    # Joint Vectors
    ax.plot(*markers[-1, [0, 1], :].T)
    ax.plot(*markers[-1, [2, 4], :].T)
    ax.plot(*markers[-1, [6, 5], :].T)
    center_shoulder, _ = get_joint(markers[-1, 1, :], markers[-1, 0, :])
    center_elbow, elbow_axis = get_joint(markers[-1, 2, :], markers[-1, 4, :])
    center_wrist, wrist_axis = get_joint(markers[-1, 5, :], markers[-1, 6, :])
    centers = np.array((center_shoulder, center_elbow, center_wrist))
    ax.plot(*centers.T, marker="o")
    arc_elbow = _plot_elbow_angle(ax, center_shoulder, center_elbow, center_wrist)
    arc_wrist = _plot_wrist_angle(ax, center_elbow, center_wrist, elbow_axis, wrist_axis)
    if export_csvs:
        base_path = os.path.join(_SAVE_PATH, "Elbow_scatter")
        _export_txt(os.path.join(base_path, "marker_pos"), markers[-1], header="x, y, z")
        _export_txt(os.path.join(base_path, "shoulder_vec"), markers[-1, [0, 1], :], header="x, y, z")
        _export_txt(os.path.join(base_path, "elbow_vec"), markers[-1, [2, 4], :], header="x, y, z")
        _export_txt(os.path.join(base_path, "wrist_vec"), markers[-1, [5, 6], :], header="x, y, z")
        _export_txt(os.path.join(base_path, "joint_centers"), centers, header="x, y, z")
        _export_txt(os.path.join(base_path, "arc_elbow"), arc_elbow, header="x, y, z")
        _export_txt(os.path.join(base_path, "arc_wrist"), arc_wrist, header="x, y, z")
        _export_txt(
            os.path.join(base_path, "fill_elbow"), np.vstack((center_elbow, arc_elbow, center_elbow)), header="x, y, z")
        _export_txt(
            os.path.join(base_path, "fill_wrist"), np.vstack((center_wrist, arc_wrist, center_wrist)), header="x, y, z")
    for i in range(markers.shape[1]):
        ax.plot(*markers[:, i, :].T, c="blue", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    plot_markers3d("recordings/05-12-24--16-36-04", 1050, 100, False)