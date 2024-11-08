import pickle
import numpy as np
from vicon_dssdk import ViconDataStream
from itertools import count

client = ViconDataStream.Client()
client.Connect('localhost')

client.EnableMarkerData()

def get_joint(point_a: tuple[tuple[float, ...], bool], point_b: tuple[tuple[float, ...], bool]):
    point_a, occluded_a = point_a
    point_b, occluded_b = point_b
    if not (occluded_a and occluded_b):
        rot_axis = np.array(point_b) - np.array(point_a)
        return np.array(point_a) + (np.array(point_b) - np.array(point_a)) * 0.5, unit_vector(rot_axis)
    return None, None

def unit_vector(vec):
    return vec / np.linalg.norm(vec)


def calc_elbow_angle(marker: dict):
    shoulder, _ = get_joint(marker["ShoulderB"], marker["ShoulderF"])
    elbow, _ = get_joint(marker["ElbowO"], marker["ElbowI"])
    wrist, _ = get_joint(marker["WristI"], marker["WristO"])
    if any([x is None for x in [shoulder, elbow, wrist]]):
        return
    vec_upper_arm = unit_vector(shoulder - elbow)
    vec_lower_arm = unit_vector(elbow - wrist)
    return np.degrees(np.arccos(np.clip(np.dot(vec_upper_arm, vec_lower_arm), -1.0, 1.0)))

def calc_wrist_angle(marker: dict):
    elbow, elbow_rot_axis = get_joint(marker["ElbowO"], marker["ElbowI"])
    wrist, wrist_rot_axis = get_joint(marker["WristI"], marker["WristO"])
    if elbow is not None and wrist is not None:
        vec_lower_arm = unit_vector(elbow - wrist)
        n1 = unit_vector(np.cross(vec_lower_arm, wrist_rot_axis))
        n2 = unit_vector(np.cross(vec_lower_arm, elbow_rot_axis))
        theta = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
        if np.dot(np.cross(n1, n2), vec_lower_arm) > 0:
            return 360 - theta
        return theta


def main(record_len: int = 0):
    data = []
    for _ in count(record_len):
        if not client.GetFrame():
            continue
        marker = dict()
        client.GetLabeledMarkers()
        for name, parent in client.GetMarkerNames("XArm"):
            marker[name] = client.GetMarkerGlobalTranslation("XArm", name)
        elbow_angle, wrist_angle = calc_elbow_angle(marker), calc_wrist_angle(marker)
        if elbow_angle is not None and wrist_angle is not None:
            print(round(elbow_angle), round(wrist_angle))
        data.append(marker)
    return data

if __name__ == "__main__":
    save_data: bool = False
    try:
        d = main()
        with open("marker_data.pkl", "wb") as file:
            pickle.dump(d, file)
    except KeyboardInterrupt:
        print("Keyboard Interrupt")



