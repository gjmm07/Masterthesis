from __future__ import annotations
import h5py
import os
import numpy as np
from typing import Sequence, List, Tuple, Literal
from operator import itemgetter
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

_LATEX_PATH = "/home/finn/Documents/LatexMA/data/"

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


def _export_txt(path: os.PathLike or str, array: np.ndarray,  *args, **kwargs):
    np.savetxt(path, array, *args, **kwargs, fmt="%f", delimiter=",", comments="")


@dataclass
class Data:
    mocap_joints: np.ndarray
    emg_data: np.ndarray
    marker_pred: np.ndarray
    marker: np.ndarray
    marker_labels: list[str]
    upper_motor_angle: np.ndarray = None
    lower_motor_angle: np.ndarray = None
    load_cell: np.ndarray = None

    imp_marker_idx = (0, 1, 2, 4, 5, 6) # important markers - check for both mocap models

    def __post_init__(self):
        self.input_data()

    def __repr__(self):
        return "Data"

    @property
    def mocap(self):
        return dict(zip(("mocap_joints", "marker", "marker_pred"),
                (self.mocap_joints, self.marker, self.marker_pred)))

    @property
    def no_mocap(self):
        return dict(zip(("emg_data", "upper_motor_angle", "lower_motor_angle", "load_cell"),
                (self.emg_data, self.upper_motor_angle, self.lower_motor_angle, self.load_cell)))

    @property
    def _time(self):
        return self.mocap_joints.shape[0] / 100

    def _plot_emg(self, ax1:plt.Axes, ax2: plt.Axes):
        t = np.linspace(0, self._time, self.emg_data.shape[0])
        ax1.plot(t, self.emg_data[:, :4] + np.atleast_2d(np.linspace(0, 0.6, 4)), lw=0.2)
        ax1.set_title("Quattro Sensor 1")
        ax2.plot(t, self.emg_data[:, 4:] + np.atleast_2d(np.linspace(0, 0.6, 4)), lw=0.2)
        ax2.set_title("Quattro Sensor 3")

    def _plot_mocap_joints(self, ax: plt.Axes):
        t = np.linspace(0, self._time, self.mocap_joints.shape[0])
        ax.set_title("MoCap Joints")
        ax.plot(t, self.mocap_joints, label=("Upper Arm", "Lower Arm"))
        ax.legend()

    def _plot_ort_angles(self, ax: plt.Axes):
        ax.set_title("Exoskeleton Angles")
        if self.upper_motor_angle is None or self.lower_motor_angle is None:
            return
        t = np.linspace(0, self._time, self.upper_motor_angle.shape[0])
        ax.plot(t, self.upper_motor_angle)
        t = np.linspace(0, self._time, self.lower_motor_angle.shape[0])
        ax.plot(t, self.lower_motor_angle)

    def _plot_load_cell(self, ax: plt.Axes):
        ax.set_title("Load Cell")
        if self.load_cell is None:
            return
        t = np.linspace(0, self._time, self.load_cell.shape[0])
        ax.plot(t, self.load_cell)

    def _plot_markers(self, markers: tuple[str, ...], axs: Sequence[plt.Axes], ref_marker: str = "ShoulderB"):
        if len(markers) != len(axs):
            raise ValueError("Markers needs same length as axes")
        indices = [self.marker_labels.index(x) for x in markers]
        data = self.marker[:, indices, :]
        ref_index = self.marker_labels.index(ref_marker)
        data -= self.marker[:, [ref_index], :]
        t = np.linspace(0, self._time, self.marker.shape[0])
        for i, (ax, title) in enumerate(zip(axs, [self.marker_labels[j] for j in indices])):
            ax.set_title(title)
            ax.plot(t, data[:, i, :], label=("x", "y", "z"))
            ax.legend()

    def plot(self, *,
             plot_emg: bool = False,
             plot_mocap_joints: bool=False,
             plot_ort_angles: bool = False,
             plot_load_cell: bool = False,
             plot_markers: bool = False,
             markers: tuple[str, ...] = (),
             ref_marker: str = ""):
        """
        Plots the current Data Instance
        :param plot_emg:            Whether to plt emg
        :param plot_mocap_joints:   Whether to plot mocap
        :param plot_ort_angles:     Whether to plot orthosis angles
        :param plot_load_cell:      Whether to plot load cell
        :param plot_markers:        Whether to plot markers
        :param markers:             Tuple of marker which are to plot, select from: ShoulderB, ShoulderF etc.
        :param ref_marker:          Label of marker which should be used as ref, e.g. ShoulderB
        """
        n_plots = sum((plot_mocap_joints, plot_ort_angles, plot_load_cell))
        if plot_emg:
            n_plots += 2
        if plot_markers:
            n_plots += len(markers)
        if not n_plots:
            return
        fig, axs = plt.subplots(n_plots, sharex=True, squeeze=False)
        axs = iter(axs.flatten())
        if plot_emg:
            self._plot_emg(next(axs), next(axs))
        if plot_mocap_joints:
            self._plot_mocap_joints(next(axs))
        if plot_ort_angles:
            self._plot_ort_angles(next(axs))
        if plot_load_cell:
            self._plot_load_cell(next(axs))
        if plot_markers:
            self._plot_markers(markers, [next(axs) for _ in markers], ref_marker)
        fig.tight_layout()
        plt.show()

    @staticmethod
    def _crop_by_percent(ary: np.ndarray, percent: np.ndarray):
        if ary is None:
            return
        bounds = (percent * ary.shape[0]).astype(int)
        return ary[bounds[0]:bounds[1]]

    def crop_data(self, *time):
        if len(time) > 2:
            return
        if len(time) == 1:
            start_time, end_time = 0, time[0]
        elif time[1] is None:
            start_time, end_time = time[0], self._time
        else:
            start_time, end_time = time
        print(start_time, end_time)
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time")
        if (end_time - start_time) > self._time:
            raise ValueError("Selected Sequence is too small for data")
        percent_bounds = np.array((start_time / self._time, end_time / self._time))
        bounds = (percent_bounds * self.mocap_joints.shape[0]).astype(int)
        self.mocap_joints = self.mocap_joints[bounds[0]:bounds[1]]
        self.marker = self.marker[bounds[0]:bounds[1]]
        self.marker_pred = self.marker_pred[bounds[0]:bounds[1]]
        self.emg_data = self._crop_by_percent(self.emg_data, percent_bounds)
        self.upper_motor_angle = self._crop_by_percent(self.upper_motor_angle, percent_bounds)
        self.lower_motor_angle = self._crop_by_percent(self.lower_motor_angle, percent_bounds)
        self.load_cell = self._crop_by_percent(self.load_cell, percent_bounds)

    def export_plot_data(self):
        _export_txt(os.path.join(_LATEX_PATH, "MoCap Joints [deg]"),
                    np.c_[np.linspace(0, self._time, self.mocap_joints.shape[0]), self.mocap_joints],
                    header="t, y, z")
        emg_data = self.emg_data[::50, :] + np.tile(np.linspace(0, 0.6, 4), 2)
        _export_txt(os.path.join(_LATEX_PATH, "Quattro Sensor st 1"),
                    np.c_[np.linspace(0, self._time, emg_data.shape[0]), emg_data[:, :4]],
                    header="t, a, b, c, d")
        _export_txt(os.path.join(_LATEX_PATH, "Quattro Sensor st 3"),
                    np.c_[np.linspace(0, self._time, emg_data.shape[0]), emg_data[:, 4:]],
                    header="t, a, b, c, d")
        _export_txt(os.path.join(_LATEX_PATH, "ort_lower"),
                    np.c_[np.linspace(0, self._time, self.lower_motor_angle.shape[0]), self.lower_motor_angle],
                    header="t, y")
        _export_txt(os.path.join(_LATEX_PATH, "ort_upper"),
                    np.c_[np.linspace(0, self._time, self.upper_motor_angle.shape[0]), self.upper_motor_angle],
                    header="t, y")
        # todo: Export marker prediction and marker if needed

    def fill(self, max_fill_time: float or Sequence[float] = 1.0):
        """
        Fills mocap joints linear if joint angles could not be calculated because marker and ref markers
        where occluded
        todo: Could be extended to joint angle where important marker where predicted (?)
        :param max_fill_time: if gap is larger than max_fill_time, gap will NOT be filled
        :return:
        """
        if type(max_fill_time) == float:
            max_fill_time = [max_fill_time] * self.mocap_joints.shape[1]
        max_fill_time = [int(x * 100) for x in max_fill_time]  # convert to samples
        for i, t_lim in enumerate(max_fill_time):
            predicted = (self.mocap_joints == -999)[:, i]
            sections = np.array(
                [0,
                 *np.repeat(np.where(predicted[1:] ^ predicted[:-1])[0], 2),
                 predicted.shape[0] - 1]).reshape(-1, 2)
            mask = np.zeros_like(self.mocap_joints[:, 0]).astype(bool)
            for bounds in sections[(predicted[sections[:, 1]]) & ((sections[:, 1] - sections[:, 0]) < t_lim), :]:
                mask[bounds[0]+1:bounds[1]+1] = True
            self.mocap_joints[mask, i] = np.interp(np.where(mask)[0], np.where(~mask)[0], self.mocap_joints[~mask, i])

    @staticmethod
    def _generate_steps(current: float, step: float, total: int):
        while current < 1:
            abs_current = round(current * total)
            yield abs_current
            current += step

    def moving_average_filter(self, filter_period: int, on: Literal["joints", "emg"]):
        if on == "joints":
            self.mocap_joints = uniform_filter1d(self.mocap_joints, size=filter_period, axis=0)
        if on == "emg":
            self.emg_data = uniform_filter1d(np.abs(self.emg_data), size=filter_period, axis=0)


    def input_data(self):
        """
        Updates the marker position, marker prediction and marker labels if the old mocap model was used
        :return:
        """
        if self.marker.shape[1] == 9:
            self.marker[:, [5, 6]] = self.marker[:, [6, 5]]
            self.marker = np.insert(self.marker,[7, 9], np.nan, axis=1)
        if self.marker_pred.shape[1] == 9:
            self.marker_pred[:, [5, 6]] = self.marker_pred[:, [6, 5]]
            self.marker_pred = np.insert(self.marker_pred,[7, 9], -999, axis=1)
        if len(self.marker_labels) == 9:
            self.marker_labels = ["ShoulderB",
                                  "ShoulderF",
                                  "ElbowO",
                                  "UpperArm",
                                  "ElbowI",
                                  "WristO",
                                  "WristI",
                                  "LowerArm1",
                                  "LowerArm2",
                                  "WristL",
                                  "Hand"]

    def get_data(self,
                 every: int,
                 tail_mocap: int,
                 tail_emg: int,
                 mocap_future: int = 0,
                 *,
                 tail_upper_mot: int = -1,
                 tail_lower_mot: int = -1,
                 tail_load_cell: int = -1
                 ):
        """
        Gets windowed data from the Data-Instance. Inspect points are set by MoCap data and the every-parameter.
        for each data source a tail can be set to return windows from the inspect point - tail to inspect point.
        Use this function to generate training and testing data for ML models
        Caution:  Due to setting of different tails, data is not in-sync anymore
        :param every:           every iteration the pointer will be set <every> samples base on mocap forward
        :param tail_mocap:      from the current pointer position, return the mocap tail
        :param tail_emg:        same for emg-data
        :param mocap_future:    samples mocap data is ahead of other
        :param tail_upper_mot:  same as tail_mocap for angle of upper mot: works only if upper mot data is present
        :param tail_lower_mot:  same as tail_mocap for angle of lower mot: works only if lower mot data is present
        :param tail_load_cell:  same as tail mocap for angle of load cell: works only if load cell data is present
        :return: numpy arrays int the form of mocap_joints, marker, marker_prediction, emg_data <opt: upper mot angle>, <opt:lower mot angle>, <opt: load cell>

        """
        step = every / self.mocap_joints.shape[0]
        start = max(tail_mocap / self.mocap_joints.shape[0],
                    tail_emg / self.emg_data.shape[0],
                    -1 if self.lower_motor_angle is None else tail_lower_mot / self.lower_motor_angle.shape[0],
                    -1 if self.upper_motor_angle is None else tail_upper_mot / self.upper_motor_angle.shape[0],
                    -1 if self.load_cell is None else tail_load_cell / self.load_cell.shape[0])
        gens = [self._generate_steps(start + mocap_future / self.mocap_joints.shape[0], step, self.mocap_joints.shape[0]),
                self._generate_steps(start, step, self.emg_data.shape[0])]
        data_sources = [(self.mocap_joints, self.marker, self.marker_pred),
                        (self.emg_data, )]
        tails = [tail_mocap, tail_emg]
        data_sinks: List[Tuple[List[np.ndarray], ...]] = [([], [], []), ([], )]

        if self.upper_motor_angle is not None and tail_upper_mot >= 1:
            gens.append(self._generate_steps(start, step, self.upper_motor_angle.shape[0]))
            data_sources.append((self.upper_motor_angle, ))
            tails.append(tail_upper_mot)
            data_sinks.append(([], ))
        if self.lower_motor_angle is not None and tail_lower_mot >= 1:
            gens.append(self._generate_steps(start, step, self.lower_motor_angle.shape[0]))
            data_sources.append((self.lower_motor_angle, ))
            tails.append(tail_lower_mot)
            data_sinks.append(([], ))
        if self.load_cell is not None and tail_load_cell >= 1:
            gens.append(self._generate_steps(start, step, self.load_cell.shape[0]))
            data_sources.append((self.load_cell, ))
            tails.append(tail_load_cell)
            data_sinks.append(([], ))
        finished = False
        while not finished:
            for gen, sources, sinks, tail in zip(gens, data_sources, data_sinks, tails):
                try:
                    stop = next(gen)
                    start = stop - tail
                    for source, sink in zip(sources, sinks):
                        sink.append(source[start:stop])
                except StopIteration:
                    finished = True
        return [np.array(sink) for sinks in data_sinks for sink in sinks]


def read_dataset(
        subject: str,
        *,
        timestamp: str or None = None,
        read_mocap: bool = True,
        read_emg: bool = True,
        read_ort: bool = False
) -> list[Data]:
    """
    Reads a complete dataset
    :param subject:
    :param timestamp:
    :param read_mocap:
    :param read_emg:
    :param read_ort:
    :return:
    """
    # todo: Handle cases where no data is present
    data = []
    for dir_ in os.listdir("recordings"):
        if timestamp is not None and dir_[:len(timestamp)] != timestamp:
            continue
        path = os.path.join("recordings", dir_)
        if read_subject(path) != subject:
            continue
        emg_data, marker, marker_pred, mocap_joints = None, None, None, None
        up_mot_angle, low_mot_angle, load_cell, marker_labels = None, None, None, None
        valid: bool = True
        if read_mocap:
            marker_pred, marker = itemgetter(
                "marker_prediction", "mocap")(read_mocap_marker(path))
            mocap_joints = read_mocap_joints(path)["joints"]
            if marker_labels is None:
                marker_labels = [x[0] for x in read_mocap_setup(path)["Markers"]]
            if not any((marker_pred.size, marker.size, mocap_joints.size, marker_labels)):
                valid = False
        if read_emg:
            emg_data = read_emg_data(path)
            if emg_data is not None:
                emg_data = np.array([x.flatten() for y in list(emg_data.values()) for x in y.values()]).T
            if emg_data is None or not emg_data.size:
                valid = False
        if read_ort:
            up_mot_angle, low_mot_angle, load_cell = itemgetter(
                "angle_upper_arm", "angle_lower_arm", "load_cell")(read_ort_data(path))
        if valid:
            data.append(
                Data(mocap_joints, emg_data, marker_pred, marker, marker_labels, up_mot_angle, low_mot_angle, load_cell))
    return data


def _crop_data(data: np.ndarray, bounds: np.ndarray):
    bounds = (bounds * data.shape[0]).astype(int)
    return data[bounds[0]:bounds[1]]


def drop_useless(data: list[Data]) -> list[Data]:
    """
    This function cuts data apart if mocap joint angle shows a very high deviation -> most probably markers were swapped
    :param data:    List of Data instances
    return:         Same size or larger list of Data instances
    """
    x = []
    for n_data in data:
        mask = (
            # np.any(np.sqrt(
            #     np.sum((n_data.marker[1:, :, :] - n_data.marker[:-1,:,:])**2, axis=2))[:, n_data.imp_marker_idx] > 50, axis=1) |
            # np.any(n_data.marker_pred[:-1] == 0, axis=1) |
            np.any(np.diff(n_data.mocap_joints, axis=0) > 20, axis=1))
        mask = [0] + list(np.where(mask)[0]) + [n_data.mocap_joints.shape[0]]
        for i, diff in enumerate(np.diff(mask)):
            new_data = {}
            if diff > 1000:
                bounds = (mask[i] + 5, mask[i + 1] - 5)
                for label, mocap in n_data.mocap.items():
                    new_data[label] = mocap[bounds[0] + 5: bounds[1] - 5]
                # handle other sample rates
                bounds = np.array(bounds) / n_data.mocap_joints.shape[0]
                for label, no_mocap in n_data.no_mocap.items():
                    if no_mocap is None:
                        new_data[label] = None
                        continue
                    new_data[label] = _crop_data(no_mocap, bounds)
                x.append(Data(**new_data, marker_labels=n_data.marker_labels))
    return x


if __name__ == "__main__":
    d = read_dataset("Finn",
                     timestamp="10-12-24--16-52-09",
                     read_ort=False)
    for nd in d:
        nd.fill()
    d = drop_useless(d)
    for nd in d:
        nd.moving_average_filter(10, on="joints")
        nd.moving_average_filter(30, on="emg")
    d[3].plot(plot_emg=True, plot_mocap_joints=True)


