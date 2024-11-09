from vicon_dssdk import ViconDataStream
from collections import deque
import asyncio
import numpy as np

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
    if not (occluded_a and occluded_b):
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
    vec_lower_arm = _unit_vector(elbow - wrist)
    n1 = _unit_vector(np.cross(vec_lower_arm, wrist_axis))
    n2 = _unit_vector(np.cross(vec_lower_arm, elbow_axis))
    theta = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
    if np.dot(np.cross(n1, n2), vec_lower_arm) > 0:
        return 360 - theta
    return theta


class ViconMoCap:

    def __init__(self, cur_pos: deque[tuple[float, float]], stop_event: asyncio.Event):
        self._cur_pos = cur_pos
        self._stop_event = stop_event

        self._client = ViconDataStream.Client()
        self._client.Connect('localhost')
        self._client.EnableMarkerData()

    def __await__(self):
        while True:
            if self._client.GetFrame():
                break
        marker = dict()
        self._client.GetLabeledMarkers()
        marker = dict()
        for name, parent in self._client.GetMarkerNames("XArm"):
            marker[name] = self._client.GetMarkerGlobalTranslation("XArm", name)
        shoulder, _ = _get_joint(marker["ShoulderB"], marker["ShoulderF"])
        elbow, elbow_axis = _get_joint(marker["ElbowO"], marker["ElbowI"])
        wrist, wrist_axis = _get_joint(marker["WristI"], marker["WristO"])
        self._cur_pos.append((
            _calc_elbow_angle(shoulder, elbow, wrist),
            _calc_wrist_angle(elbow, wrist, elbow_axis, wrist_axis)
        ))
        if not self._stop_event.is_set():
            yield from asyncio.sleep(0.5).__await__()
            yield from self.__await__()

    def stop(self):
        pass


