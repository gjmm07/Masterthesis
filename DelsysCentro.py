from pythonnet import load
from delsys_secrets import key, license
import numpy as np

load("coreclr")
import clr

clr.AddReference("resources\\DelsysAPI")
clr.AddReference("System.Collections")

from Aero import AeroPy


class DelsysCentro:

    def __init__(self):
        self._base = AeroPy()
        self._keys = []
        self._sensors = []

    def _connect(self):
        self._base.ValidateBase(key, license)

    def _save_sensor_setup(self):
        for sensor in self._sensors:
            print(dir(sensor))
            print(f"{sensor.PairNumber} "
                  f"{sensor.FriendlyName} {sensor.Configuration.ModeString} {sensor.PairNumber} {sensor.Id}")
            for channel in sensor.TrignoChannels:
                print(f"\t{channel.Name} using {channel.SampleRate} Hz")
                print(dir(channel))

    def _scan(self):
        _ = self._base.ScanSensors().Result
        self._sensors = self._base.GetScannedSensorsFound()
        for sensor in self._sensors:
            for channel in sensor.TrignoChannels:
                self._keys.append(channel.Id)
        # self._save_sensor_setup()
        self._base.SelectAllSensors()

    def get_data(self):
        """
        :return: a EMG (IMU) dataframe
        """
        while True:
            if not self._base.CheckDataQueue():
                continue
            raw_data = self._base.PollData()
            data = []
            for emg_key in self._keys:
                data.append(np.array(raw_data[emg_key]))
            return data

    def _start_station(self):
        if self._get_pipeline_state() == "Connected":
            self._base.Configure()
        self._base.Start()

    def _stop_station(self):
        self._base.Stop()

    def _get_pipeline_state(self):
        return self._base.GetPipelineState()

    def __enter__(self):
        self._connect()
        self._scan()
        self._start_station()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")
        self._stop_station()





