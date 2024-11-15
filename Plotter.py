import matplotlib.pyplot as plt
import numpy as np
import utils
import os
import h5py


joint_keys, joint_data = utils.read_data("recordings/15-11-24--12-07-41/MoCap/joint_data.h5")
# ort_keys, ort_data = utils.read_data("recordings/15-11-24--10-40-10/Ort_data/data.h5")
emg_keys, emg_data = utils.read_data("recordings/15-11-24--12-07-41/EMG_data/Avanti Sensor st 5/data.h5")
marker_keys, marker_data = utils.read_data("recordings/15-11-24--12-07-41/MoCap/marker.h5")

marker_data = marker_data[0]

print(marker_data.shape)
print(np.all(marker_data == 0, axis=2))
exit()
joint_data = joint_data[0]
print(joint_data.shape)
# joint_data[joint_data < -900] = np.nan
#
fig, axs = plt.subplots(3, 1)
# axs[0].plot(joint_data[:, 0])
# axs[0].plot(joint_data[:, 1])
#
# print(ort_keys)
# axs[1].plot(ort_data[0])
# axs[1].plot(ort_data[1])
#
axs[2].plot(emg_data[0])
plt.show()
