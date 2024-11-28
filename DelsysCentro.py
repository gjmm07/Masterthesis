import os
from pythonnet import load
from delsys_secrets import key, license
import numpy as np
import time
from scipy.signal import butter, lfilter, iirnotch, sosfilt

load("coreclr")
import clr

clr.AddReference("resources\\DelsysAPI")
clr.AddReference("System.Collections")

from Aero import AeroPy
from DelsysAPI import Exceptions
from DelsysAPI.Components import TrignoRf

_STICKER_LOOKUP = {"000132f0-0010-0000-0000-000000000000": 1,
                   "00013120-001b-0000-0000-000000000000": 2,
                   "00013094-001b-0000-0000-000000000000": 3,
                   "0001391a-000e-0000-0000-000000000000": 5,
                   "000131ab-0010-0000-0000-000000000000": 3}

_APPLY_FILTER = True

class _DelsysFilter:
    """
    Remove DC-Offset from data
    """
    def __init__(self, sensors: list[TrignoRf]):
        self._butter_params = []
        self._z_butter = []
        for sensor in sensors:
            for channel in sensor.TrignoChannels:
                self._butter_params.append(
                    butter(2 if "EMG" in channel.Name else 0, 1, fs=channel.SampleRate, btype="high", output="sos"))
                self._z_butter.append(np.zeros((1, 2)))
        self._counter: int = 0
        self._ready: bool = False

    def highpass_filter(self, data_bundle: list[list[float]]) -> list[list[float]]:
        results = []
        for i, (data, butter_parm) in enumerate(zip(data_bundle, self._butter_params)):
            res, self._z_butter[i] = sosfilt(butter_parm, data, zi=self._z_butter[i])
            results.append(list(res))
        if self._counter < 30:
            self._counter += 1
            return [list(np.zeros_like(x)) for x in results]
        self._ready = True
        return results

    @property
    def is_ready(self):
        if _APPLY_FILTER:
            return self._ready
        return True


class DelsysCentro:

    def __init__(self):
        self._base = AeroPy()
        self._keys = []
        self._sensors: list[TrignoRf] = []
        self._filters: _DelsysFilter or None = None

    @property
    def sensors(self) -> list[str]:
        return [f"{sensor.FriendlyName} st {_STICKER_LOOKUP.get(str(sensor.Id))}" for sensor in self._sensors]

    @property
    def channels(self) -> list[list[str]]:
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

    def pair_sensor(self):
        self._base.PairSensor()
        while self._base.CheckPairStatus():
            continue
        print()

    def scan(self):
        _ = self._base.ScanSensors().Result
        self._sensors = self._base.GetScannedSensorsFound()
        for sensor in self._sensors:
            for channel in sensor.TrignoChannels:
                self._keys.append(channel.Id)
        self._base.SelectAllSensors()
        self._filters = _DelsysFilter(self._sensors)

    def print_possible_modes(self):
        if not self._sensors:
            print("scan for sensors first")
            return
        for i, sensor in enumerate(self._sensors):
            print(f"{sensor.FriendlyName} with sticker {_STICKER_LOOKUP.get(str(sensor.Id))}")
            for j, mode in enumerate(self._base.AvailibleSensorModes(i)):
                print("\t", j, mode)

    def _set_preferred_mode(self):
        """
        Sets sensors to my preferred sensor mode: EMG raw 20-450HZ bandpass +/- 5,5mV ~2000 Hz sample rate
        :return:
        """
        if not self._sensors:
            print("scan for sensors first")
            return
        sensor_mode = {
            "Avanti Sensor": 4,
            "Duo Sensor": 103,
            "Quattro Sensor": 72
        }
        for i, sensor in enumerate(self._sensors):
            modes = self._base.AvailibleSensorModes(i)
            self._base.SetSampleMode(i, modes[sensor_mode[sensor.FriendlyName]])
        self._update_keys()
        self._filters = _DelsysFilter(self._sensors)

    def _update_keys(self):
        """
        Updates the keys to grab data from sample dict
        :return:
        """
        if not self._sensors:
            print("scan for sensors first")
            return
        self._keys = []
        for sensor in self._sensors:
            for channel in sensor.TrignoChannels:
                self._keys.append(channel.Id)

    def get_data(self):
        data = self._get_data()
        return data

    def _get_data(self) -> list[list[float]] | None:
        """
        :return: a EMG (IMU) dataframe
        """
        if self._get_pipeline_state() != "Running":
            return
        while True:
            if not self._base.CheckDataQueue():
                time.sleep(0.01)
                continue
            raw_data = self._base.PollData()
            data = []
            for emg_key in self._keys:
                data.append(list(raw_data[emg_key]))
            if _APPLY_FILTER:
                return self._filters.highpass_filter(data)
            return data

    def start_station(self):
        if self._get_pipeline_state() == "Connected":
            self._base.Configure()
        self._base.Start()

    def _stop_station(self):
        self._base.Stop()

    def _get_pipeline_state(self):
        return self._base.GetPipelineState()

    @property
    def is_ready(self):
        return self._filters.is_ready

    def __enter__(self):
        if not self._connect():
            return
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")
        self._stop_station()

if __name__ == "__main__":
    with DelsysCentro() as dcentro:
        dcentro.pair_sensor()




