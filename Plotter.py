import matplotlib.pyplot as plt
import numpy as np
import utils
import os
import h5py


joint_keys, joint_data = utils.read_data("recordings/18-11-24--17-42-35/MoCap/joint_data.h5")
ort_keys, ort_data = utils.read_data("recordings/18-11-24--17-42-35/Ort_data/data.h5")
emg_keys, emg_data = utils.read_data("recordings/18-11-24--17-42-35/EMG_data/Avanti Sensor st 5/data.h5")
marker_keys, marker_data = utils.read_data("recordings/18-11-24--17-42-35/MoCap/marker.h5")


fig, axs = plt.subplots(3, 1)
axs[0].plot(joint_data[0])

axs[1].plot(ort_data[1])
axs[1].plot(ort_data[0])

axs[2].plot(emg_data[0])

plt.show()