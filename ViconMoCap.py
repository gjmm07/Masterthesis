import os
from PIL.ImagePalette import random
from vicon_dssdk import ViconDataStream
from collections import deque
import asyncio
import numpy as np
import h5py
import utils
from scipy.optimize import root, minimize
from typing import Sequence, Optional


def _get_brother_markers(markers: np.ndarray):
    """
    Gets the best (marker which distance doesn't change much)  markers for every marker
    :return:
    """
    best_brothers, mean_dists = [], []
    for mask in np.eye(markers.shape[1]).astype(bool):
        marker = markers[:, mask]
        dists = np.linalg.norm(markers - marker, axis=2)
        mean_dist = np.mean(dists, axis=0)
        std_dists = np.std(dists, axis=0)
        order = np.argsort(std_dists)
        best_brothers.append(order)
        mean_dists.append(mean_dist[order])
    return np.array(best_brothers)[:, 1:], np.array(mean_dists)[:, 1:]

def _mask_occluded_samples(markers: np.ndarray):
    return ~np.any(~np.all(markers, axis=2), axis=1)


class _MarkerPrediction:

    def __init__(self,
                 collect_only: bool,
                 best_brothers: Optional[np.ndarray] = None,
                 mean_distances: Optional[np.ndarray] = None):
        self._collect_only = collect_only
        if not collect_only and (best_brothers is None or mean_distances is None):
            raise ValueError("If you don't want to collect only, you need to specify best_brothers and mean_distance")
        if not collect_only:
            self._best_brothers = best_brothers
            self._mean_distances = mean_distances
            self._last_valid_position = np.zeros((self._mean_distances.shape[0], 3))
            self._last_seen = np.zeros(self._mean_distances.shape[0]) - 1
            self._initial_guess = np.zeros((self._mean_distances.shape[0], 3))

    @classmethod
    def from_file_data(cls, path: os.PathLike or str):
        keys, marker_pos = utils.read_data(path)
        marker_pos = marker_pos[keys.index("mocap")]
        mask = _mask_occluded_samples(marker_pos)
        min_frames = 100
        if np.sum(mask) < min_frames:
            raise ValueError(
                f"Calculating model failed because not all markers where visible in at least {min_frames} frames, "
                f"only {np.sum(mask)} valid frames")
        best_brothers, mean_distances = _get_brother_markers(marker_pos[mask])
        return cls(False, best_brothers, mean_distances)

    @classmethod
    def from_file(cls, path: os.PathLike or str = "marker_model"):
        best_brothers = np.genfromtxt(os.path.join(path, "best_brothers.csv"), delimiter=",", dtype=int)
        mean_distance = np.genfromtxt(os.path.join(path, "mean_distances.csv"), delimiter=",")
        return cls(False, best_brothers, mean_distance)

    def save_model(self, path: os.PathLike or str = "marker_model/"):
        os.makedirs(path, exist_ok=True)
        np.savetxt(os.path.join(path, "best_brothers.csv"), self._best_brothers, fmt="%i", delimiter=",")
        np.savetxt(os.path.join(path, "mean_distances.csv"), self._mean_distances, fmt="%1.5f", delimiter=",")

    def test(self, markers: np.ndarray):
        """
        tests the algorithm
        :param              markers: marker array to be tested - ensure marker are sequential and none is occluded
        :return:
        """
        pred_index = 5
        n_predictions = 10
        from_ = np.random.randint(2, markers.shape[0] - n_predictions)
        to = from_ + n_predictions
        for i, marker in enumerate(markers):
            occluded = np.zeros(markers.shape[1]).astype(bool)
            if from_ <= i <= to:
                occluded[pred_index] = True
                sim_marker = marker.copy()
                sim_marker[pred_index] = np.zeros((3, ))
                pred, pred_type = self.predict(sim_marker, occluded, (pred_index, ))
                print(np.linalg.norm(pred[pred_index] - marker[pred_index]))
            else:
                self.predict(marker, occluded, (pred_index, ))


    def predict(
            self, markers: np.ndarray, occluded: np.ndarray, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        predict marker position
        :param              markers: array containing marker positions
        :param              occluded: array containing if a marker is occluded
        :param              indices: indices of markers that are to predict
        :return:            marker array with if needed predicted marker positions
                            array containing prediction typ:
                                -2:     no need to predict as marker is only for redundancy
                                -1:     predicted
                                0:      unable to predict or occluded
                                1:      not occluded
        """
        # todo: Kalman filter using last seen position
        if self._collect_only:
            return markers, (~occluded).astype(int)
        self._last_valid_position[~occluded] = markers[~occluded]
        self._last_seen = np.where(occluded, self._last_seen + 1, 0)
        self._initial_guess[~occluded] = markers[~occluded]
        indices = list(indices)
        predicted = []
        for i, occ in enumerate(occluded):
            if not occ:
                predicted.append(1)
                if indices and indices[0] == i:
                    del indices[0]
                continue
            elif i not in indices:
                # no need to predict as marker is not used
                predicted.append(-2)
                continue
            bros = self._best_brothers[i, :3]
            if any(occluded[bros]):
                # unable to predict because bros are also occluded
                predicted.append(0)
                continue
            dists = self._mean_distances[i, :3]
            points = markers[bros, :]
            pred = self._triangulate(self._initial_guess[i], points, dists)
            markers[i] = pred
            self._initial_guess[i] = pred
            predicted.append(-1)
        return markers, np.array(predicted)

    @staticmethod
    def _triangulate(initial_guess: np.ndarray, points, dists):

        def funcs(args):
            eqs = np.sum((args - points) ** 2, axis=1) - dists ** 2
            return eqs

        sol = root(funcs, initial_guess)
        return sol.x


def _get_joint(
        point_a: np.ndarray,
        point_b: np.ndarray) -> tuple[np.ndarray, ...] | tuple[None, None]:
    """
    Gets the joint center position (x, y, z) as well as the axis of rotation vec[dim=3] from a single joint
    :param point_a: first marker position belonging to the joint
    :param point_b: second marker position belonging to the same join
    :return: tuple
    """
    rot_axis = np.array(point_b) - np.array(point_a)
    return np.array(point_a) + (np.array(point_b) - np.array(point_a)) * 0.5, _unit_vector(rot_axis)

def _unit_vector(vec: np.ndarray):
    return vec / np.linalg.norm(vec)


def _calc_elbow_angle(
        shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    # if any([x is None for x in [shoulder, elbow, wrist]]):
    #     return
    vec_upper_arm = _unit_vector(shoulder - elbow)
    vec_lower_arm = _unit_vector(elbow - wrist)
    return np.degrees(np.arccos(np.clip(np.dot(vec_upper_arm, vec_lower_arm), -1.0, 1.0)))


def _calc_wrist_angle(
        elbow: np.ndarray, wrist: np.ndarray, elbow_axis: np.ndarray, wrist_axis: np.ndarray) -> float:
    # if elbow is None or wrist is None:
    #     return
    vec_lower_arm = _unit_vector(wrist - elbow)
    n1 = _unit_vector(np.cross(wrist_axis, vec_lower_arm))
    n2 = _unit_vector(np.cross(vec_lower_arm, elbow_axis))
    theta = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
    return theta

class _ViconMoCap:

    def __init__(self,
                 enable_marker: bool,
                 enable_segments: bool,
                 cur_pos: deque[tuple[float, float]],
                 stop_event: asyncio.Event,
                 path: os.PathLike or str,
                 start_recoding: asyncio.Event,
                 stop_recording: asyncio.Event,
                 save_every: int = 20
                 ):
        self._client = ViconDataStream.Client()
        self._client.Connect('localhost')
        if enable_marker:
            self._client.EnableMarkerData()
        if enable_segments:
            self._client.EnableSegmentData()

        self._cur_pos = cur_pos
        self._stop_event = stop_event

        self._zero_pos: tuple[float, float] = (0, 0)
        self._base_path = os.path.join(path, "MoCap")
        os.makedirs(self._base_path, exist_ok=True)
        self._data_path = os.path.join(self._base_path, "joint_data.h5")
        self._start_recording: asyncio.Event = start_recoding
        self._stop_recording: asyncio.Event = stop_recording
        self._save_every = save_every
        self._create_joint_file()

    def _create_joint_file(self):
        with h5py.File(self._data_path, "w") as file:
            file.create_dataset("joints",
                                shape=(0, 2),
                                maxshape=(None, 2),
                                dtype="float32",
                                chunks=True)

    async def set_zero(self, delay: float = 0.0, direction: bool = False):
        await asyncio.sleep(delay)
        print("setting zero")
        cur_pos = self._cur_pos[-1]
        if direction:
            self._zero_pos = self._zero_pos[0] - cur_pos[0], self._zero_pos[1] - cur_pos[1]
        else:
            self._zero_pos = cur_pos[0] + self._zero_pos[0], cur_pos[1] + self._zero_pos[1]
        print(self._zero_pos)

    async def run(self):
        raise NotImplementedError("Not Implemented")

    async def _save_manger_joint(self, join_que: asyncio.Queue[tuple[float, float]]):
        await self._start_recording.wait()
        i = 0
        data = []
        while not self._stop_recording.is_set():
            try:
                joints = await asyncio.wait_for(join_que.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            joints = [joint if joint is not None else -999 for joint in joints]
            data.append(joints)
            if i > self._save_every:
                utils.save(self._data_path, np.array(data), "joints")
                data = []
                i = 0
            i += 1
        if data:
            utils.save(self._data_path, np.array(data), "joints")

    def stop(self):
        pass


class ViconMoCapMarker(_ViconMoCap):

    def __init__(self,
                 cur_pos: deque[tuple[float, float]],
                 stop_event: asyncio.Event,
                 path: os.PathLike or str,
                 start_recoding: asyncio.Event,
                 stop_recording: asyncio.Event,
                 save_every: int = 20,
                 marker_predictor: _MarkerPrediction or None = None):

        super().__init__(
            True, False, cur_pos, stop_event,
            path, start_recoding, stop_recording, save_every)
        self._marker_path = os.path.join(self._base_path, "marker.h5")
        with h5py.File(self._marker_path, "w") as file:
            file.create_dataset("mocap",
                                shape=(0, 9, 3),
                                maxshape=(None, 9, 3),
                                dtype="float32",
                                chunks=True)
            file.create_dataset("marker_prediction",
                                shape=(0, 9),
                                maxshape=(None, 9),
                                dtype="int",
                                chunks=True)
        self._data_bundle: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
        self._save_setup()
        self._marker_predictor = marker_predictor
        if marker_predictor is None:
            self._marker_predictor = _MarkerPrediction.from_file()
            pth = os.path.join(self._base_path, "marker_model")
            print(pth)
            os.makedirs(pth)
            self._marker_predictor.save_model(pth)
        #_MarkerPrediction.from_file_data("recordings/16-11-24--16-33-29/MoCap/marker.h5")
        # self._marker_predictor = _MarkerPrediction(np.array([1, 2, 3]), np.array([1]), np.array([1]))

    @classmethod
    def collect_only(cls,
                     stop_event: asyncio.Event,
                     path: os.PathLike or str,
                     start_recoding: asyncio.Event,
                     stop_recording: asyncio.Event,
                     save_every: int = 20
                     ):
        marker_predictor = _MarkerPrediction(True)
        return cls(deque((), maxlen=0),
                   stop_event,
                   path,
                   start_recoding,
                   stop_recording,
                   save_every,
                   marker_predictor=marker_predictor)

    def save_marker_model(self):
        mark_pred = _MarkerPrediction.from_file_data(self._marker_path)
        mark_pred.save_model()

    def _save_setup(self):
        self._client.GetFrame()
        with open(os.path.join(self._base_path, "setup.txt"), "w") as file:
            for subject in self._client.GetSubjectNames():
                file.write(subject + "\n")
                file.write("### Markers\n")
                for marker in self._client.GetMarkerNames(subject):
                    file.write(f"\t{marker}\n")
                file.write("### Segments\n")
                for segment in self._client.GetSegmentNames(subject):
                    file.write(f"\t {segment}\n")

    def _get_empty_bundle(self) -> dict:
        raise DeprecationWarning("Should not be used anymore")
        empty_data = {}
        self._client.GetFrame()
        subject = self._client.GetSubjectNames()[0]
        for marker in self._client.GetMarkerNames(subject):
            print(marker)
            empty_data[marker[0]] = []
        return empty_data

    def _save_data_marker(self):
        if not self._data_bundle[0]:
            return
        utils.save(self._marker_path,
                   [np.array(x) for x in self._data_bundle],
                   ("mocap", "marker_prediction"))
        self._data_bundle = ([], [])

    def _calc_angles(self) -> tuple[np.ndarray, np.ndarray, float or None, float or None]:
        # marker = dict()
        self._client.GetLabeledMarkers()
        positions, occluded = [], []
        for name, parent in self._client.GetMarkerNames("XArm"):
            pos, occ = self._client.GetMarkerGlobalTranslation("XArm", name)
            positions.append(pos)
            occluded.append(occ)
        # 0: ShoulderB
        # 1: ShoulderF
        # 2: ElbowO
        # 4: ElbowI
        # 5: WristI
        # 6: WristO
        positions = np.array(positions)
        occluded = np.array(occluded)
        indices = (0, 1, 2, 4, 5, 6)
        marker, predicted = self._marker_predictor.predict(positions.copy(), occluded, indices)
        if len(predicted) != 9:
            print(len(predicted))
        angles = (None, None)
        if np.all(predicted[list(indices)] != 0):
            shoulder, _ = _get_joint(marker[0], marker[1])
            elbow, elbow_axis = _get_joint(marker[2], marker[4])
            wrist, wrist_axis = _get_joint(marker[5], marker[6])
            angles = (
                _calc_elbow_angle(shoulder, elbow, wrist), _calc_wrist_angle(elbow, wrist, elbow_axis, wrist_axis))
        return marker, predicted, angles[0], angles[1]

    def _update_queue(self, calculated_pos: tuple[float, float]):
        self._cur_pos.append((calculated_pos[0] - self._zero_pos[0], calculated_pos[1] - self._zero_pos[1]))

    async def _save_manager_marker(self, queue: asyncio.Queue[tuple[np.ndarray, np.ndarray]]):
        await self._start_recording.wait()
        i = 0
        while not self._stop_recording.is_set():
            try:
                data = await asyncio.wait_for(queue.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            # self._data_bundle = {key: value + [marker[key][0]] for key, value in self._data_bundle.items()}
            self._data_bundle[0].append(data[0])
            self._data_bundle[1].append(data[1])
            if i > self._save_every:
                self._save_data_marker()
                i = 0
            i += 1
        self._save_data_marker()

    async def _get_data(
            self,
            marker_queue: asyncio.Queue[tuple[np.ndarray, np.ndarray]],
            joint_queue: asyncio.Queue[tuple[float, float]]):

        while not self._stop_event.is_set():
            await asyncio.sleep(0.01)
            if not self._client.GetFrame():
                continue
            marker, predicted, *calc_pos = self._calc_angles()
            if self._start_recording.is_set() and not self._stop_recording.is_set():
                await marker_queue.put((marker, predicted))
                await joint_queue.put(calc_pos)
            if not any((x is None for x in calc_pos)):
                self._update_queue(calc_pos)

    async def run(self):
        marker_que: asyncio.Queue[tuple[np.ndarray, np.ndarray]] = asyncio.Queue()
        joint_que: asyncio.Queue[tuple[float, float]] = asyncio.Queue()
        await asyncio.gather(
            self._get_data(marker_que, joint_que),
            self._save_manager_marker(marker_que),
            self._save_manger_joint(joint_que)
        )


class ViconMoCapSegment(_ViconMoCap):

    def __init__(self,
                 cur_pos: deque[tuple[float, float]],
                 stop_event: asyncio.Event,
                 path: os.PathLike or str,
                 start_recoding: asyncio.Event,
                 stop_recording: asyncio.Event):
        super().__init__(
            False, True, cur_pos, stop_event, path, start_recoding, stop_recording)

    async def _save_manager(self, queue: asyncio.Queue):
        await asyncio.sleep(0.1)

    async def _get_data(self, queue: asyncio.Queue):
        while not self._stop_event.is_set():
            await asyncio.sleep(0.01)
            if not self._client.GetFrame():
                continue
            euler, occ = self._client.GetSegmentLocalRotationEulerXYZ("finn", "LowArm")
            self._client.GetSegmentGlobalRotationEulerXYZ()
            cur_pos = np.degrees(euler[0]), np.degrees(euler[2])
            await queue.put(cur_pos)
            self._cur_pos.append((self._zero_pos[0] - cur_pos[0], cur_pos[1] - self._zero_pos[1]))

    async def set_zero(self, delay: float = 0.0, direction: bool = False):
        await super().set_zero(delay, True)

    async def run(self):
        que = asyncio.Queue()
        await asyncio.gather(
            self._get_data(que),
            self._save_manager(que)
        )


if __name__ == "__main__":
    mp = _MarkerPrediction.from_file_data("recordings/16-11-24--16-33-29/MoCap/marker.h5")
    mp.save_model()

    mp = _MarkerPrediction.from_file()
    # *k, (pred, m) = utils.read_data(
    #     # "recordings/16-11-24--16-33-29/MoCap/marker.h5"
    #     "recordings/16-11-24--16-34-31/MoCap/marker.h5"
    # )
    # # print(pred)
    # x = np.all(pred == -1, axis=1)
    # # print(np.where(x[1:] != x[:-1])[0])
    # mp.test(m[:94])



