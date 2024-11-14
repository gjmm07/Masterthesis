import matplotlib.pyplot as plt
import numpy as np
import utils
import os
import h5py


joint_keys, joint_data = utils.read_data("recordings/14-11-24--17-56-36/MoCap/joint_data.h5")
ort_keys, ort_data = utils.read_data("recordings/14-11-24--17-56-36/Ort_data/data.h5")

joint_data = joint_data[0]
joint_data[joint_data < -900] = np.nan

fig, axs = plt.subplots(2, 1)
axs[0].plot(joint_data[:, 0])
axs[0].plot(joint_data[:, 1])

axs[1].plot(ort_data[0])
axs[1].plot(ort_data[1])
plt.show()
