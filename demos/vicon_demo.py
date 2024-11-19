import pickle
import numpy as np
from vicon_dssdk import ViconDataStream
from itertools import count
from ViconMoCap import get_joint, _unit_vector
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


client = ViconDataStream.Client()
client.Connect('localhost')

def calc_elbow_angle(marker: dict):
    shoulder, _ = get_joint(marker["ShoulderB"], marker["ShoulderF"])
    elbow, _ = get_joint(marker["ElbowO"], marker["ElbowI"])
    wrist, _ = get_joint(marker["WristI"], marker["WristO"])
    if any([x is None for x in [shoulder, elbow, wrist]]):
        return
    vec_upper_arm = _unit_vector(shoulder - elbow)
    vec_lower_arm = _unit_vector(elbow - wrist)
    return np.degrees(np.arccos(np.clip(np.dot(vec_upper_arm, vec_lower_arm), -1.0, 1.0)))

def calc_wrist_angle(marker: dict):
    elbow, elbow_rot_axis = get_joint(marker["ElbowO"], marker["ElbowI"])
    wrist, wrist_rot_axis = get_joint(marker["WristI"], marker["WristO"])
    if elbow is not None and wrist is not None:
        vec_lower_arm = _unit_vector(elbow - wrist)
        n1 = _unit_vector(np.cross(vec_lower_arm, wrist_rot_axis))
        n2 = _unit_vector(np.cross(vec_lower_arm, elbow_rot_axis))
        theta = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
        if np.dot(np.cross(n1, n2), vec_lower_arm) > 0:
            return 360 - theta
        return theta


def main(record_len: int = 0):
    client.EnableMarkerData()
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

def main2(record_len: int = 3000):
    client.EnableSegmentData()
    data = []
    try:
        for i in range(record_len):
            if not client.GetFrame():
                continue

            global_up_arm, _ = client.GetSegmentGlobalRotationEulerXYZ("XArm", "UpArm")
            global_low_arm, _ = client.GetSegmentGlobalRotationEulerXYZ("XArm", "LowArm")
            # data.append(np.array(xx) - np.array(euler_xyz))
            data.append(np.array(global_up_arm) - np.array(global_low_arm))
    except KeyboardInterrupt:
        return
    return data


if __name__ == "__main__":
    save_data: bool = False
    try:
        d = main2()
        data = np.degrees(np.array(d)[1:, :])
        # data = np.where(data < 0, data + np.pi * 2, data)
        plt.plot(data)
        plt.legend(("local_euler_X", "local_euler_Y", "local_euler_Z"))
        plt.xlabel("Samples")
        plt.ylabel("deg")
        plt.show()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")



