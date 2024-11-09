import os
from pythonnet import load
from delsys_secrets import key, license
import numpy as np

load("coreclr")
import clr

clr.AddReference("resources\\DelsysAPI")
clr.AddReference("System.Collections")

from Aero import AeroPy
from DelsysAPI import Exceptions

_STICKER_LOOKUP = {"000132f0-0010-0000-0000-000000000000": 1,
                   "00013120-001b-0000-0000-000000000000": 2,
                   "00013094-001b-0000-0000-000000000000": 3}

class DelsysCentro:
    # todo: High pass filter to remove dc-offset?

    def __init__(self):
        self._base = AeroPy()
        self._keys = []
        self._sensors = []

    @property
    def sensors(self):
        return [f"{sensor.FriendlyName} st {_STICKER_LOOKUP.get(str(sensor.Id))}" for sensor in self._sensors]

    @property
    def channels(self):
        return [[channel.Name for channel in sensor.TrignoChannels] for sensor in self._sensors]

    def _connect(self):
        try:
            self._base.ValidateBase(key, license)
            return True
        except Exceptions.PipelineException:
            print("Delsys not connected")
            return False

    def save_sensor_setup(self, path: os.PathLike or str):
        path = os.path.join(path, "SensorSetup.txt")
        with open(path, "w") as file:
            for sensor in self._sensors:
                file.write(
                    f"{sensor.FriendlyName} with sticker "
                    f"{_STICKER_LOOKUP.get(str(sensor.Id))} in mode "
                    f"{sensor.Configuration.ModeString} \n")
                for channel in sensor.TrignoChannels:
                    file.write(f"\t{channel.Name} using {channel.SampleRate} Hz \n")

    def _scan(self):
        _ = self._base.ScanSensors().Result
        self._sensors = self._base.GetScannedSensorsFound()
        for sensor in self._sensors:
            for channel in sensor.TrignoChannels:
                self._keys.append(channel.Id)
        self._base.SelectAllSensors()

    def get_data(self) -> list[list[float]] | None:
        """
        :return: a EMG (IMU) dataframe
        """
        if self._get_pipeline_state() != "Running":
            return
        while True:
            if not self._base.CheckDataQueue():
                continue
            raw_data = self._base.PollData()
            data = []
            for emg_key in self._keys:
                data.append(list(raw_data[emg_key]))
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
        if not self._connect():
            return
        self._scan()
        self._start_station()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")
        self._stop_station()

if __name__ == "__main__":
    with DelsysCentro() as dc:
        print(dc.sensors)
        print(dc.get_data())




