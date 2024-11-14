import os
from vicon_dssdk import ViconDataStream
from collections import deque
import asyncio
import numpy as np
import h5py
import utils


def _get_joint(
        point_a: tuple[tuple[float, ...], bool],
        point_b: tuple[tuple[float, ...], bool]) -> tuple[np.ndarray, ...] | tuple[None, None]:
    """
    Gets the joint center position (x, y, z) as well as the axis of rotation vec[dim=3] from a single joint
    :param point_a: first marker position belonging to the joint
    :param point_b: second marker position belonging to the same join
    :return: tuple
    """
    point_a, occluded_a = point_a
    point_b, occluded_b = point_b
    if not occluded_a and not occluded_b:
        rot_axis = np.array(point_b) - np.array(point_a)
        return np.array(point_a) + (np.array(point_b) - np.array(point_a)) * 0.5, _unit_vector(rot_axis)
    return None, None

def _unit_vector(vec: np.ndarray):
    return vec / np.linalg.norm(vec)

def _calc_elbow_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray):
    if any([x is None for x in [shoulder, elbow, wrist]]):
        return
    vec_upper_arm = _unit_vector(shoulder - elbow)
    vec_lower_arm = _unit_vector(elbow - wrist)
    return np.degrees(np.arccos(np.clip(np.dot(vec_upper_arm, vec_lower_arm), -1.0, 1.0)))


def _calc_wrist_angle(elbow: np.ndarray, wrist: np.ndarray, elbow_axis: np.ndarray, wrist_axis: np.ndarray):
    if elbow is None or wrist is None:
        return
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
                 save_every: int = 20):

        super().__init__(True, False, cur_pos, stop_event, path, start_recoding, stop_recording,
                         save_every)
        self._marker_path = os.path.join(self._base_path, "marker.h5")
        with h5py.File(self._marker_path, "w") as file:
            file.create_dataset("mocap",
                                shape=(0, 9, 3),
                                maxshape=(None, 9, 3),
                                dtype="float32",
                                chunks=True)
        self._data_bundle: dict = self._get_empty_bundle()
        self._save_setup()

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
        empty_data = {}
        self._client.GetFrame()
        subject = self._client.GetSubjectNames()[0]
        for marker in self._client.GetMarkerNames(subject):
            print(marker)
            empty_data[marker[0]] = []
        return empty_data

    def _save_data_marker(self, data: dict):
        if not list(data.values())[0]:
            return
        data = np.array(list(data.values())).transpose(1, 0, 2)
        utils.save(self._marker_path, data, "mocap")
        self._data_bundle = {key: value.clear() or value for key, value in self._data_bundle.items()}

    def _calc_angles(self) -> tuple[dict, float, float]:
        marker = dict()
        self._client.GetLabeledMarkers()
        marker = dict()
        for name, parent in self._client.GetMarkerNames("XArm"):
            marker[name] = self._client.GetMarkerGlobalTranslation("XArm", name)
        shoulder, _ = _get_joint(marker["ShoulderB"], marker["ShoulderF"])
        elbow, elbow_axis = _get_joint(marker["ElbowO"], marker["ElbowI"])
        wrist, wrist_axis = _get_joint(marker["WristI"], marker["WristO"])
        return (marker,
                _calc_elbow_angle(shoulder, elbow, wrist),
                _calc_wrist_angle(elbow, wrist, elbow_axis, wrist_axis))

    def _update_queue(self, calculated_pos: tuple[float, float]):
        self._cur_pos.append((calculated_pos[0] - self._zero_pos[0], calculated_pos[1] - self._zero_pos[1]))

    async def _save_manager_marker(self, queue: asyncio.Queue):
        await self._start_recording.wait()
        i = 0
        while not self._stop_recording.is_set():
            try:
                marker = await asyncio.wait_for(queue.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            self._data_bundle = {key: value + [marker[key][0]] for key, value in self._data_bundle.items()}
            if i > self._save_every:
                self._save_data_marker(self._data_bundle)
                i = 0
        self._save_data_marker(self._data_bundle)

    async def _get_data(self, marker_queue: asyncio.Queue[dict], joint_queue: asyncio.Queue[tuple[float, float]]):
        while not self._stop_event.is_set():
            await asyncio.sleep(0.01)
            if not self._client.GetFrame():
                continue
            marker, *calc_pos = self._calc_angles()
            if self._start_recording.is_set() and not self._stop_recording.is_set():
                await marker_queue.put(marker)
                await joint_queue.put(calc_pos)
            if not any((x is None for x in calc_pos)):
                self._update_queue(calc_pos)

    async def run(self):
        marker_que: asyncio.Queue[dict] = asyncio.Queue()
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





