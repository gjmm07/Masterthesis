import asyncio
import os
import time
from multiprocessing import Process, Event, Lock
from DelsysCentro import DelsysCentro
import h5py
import numpy as np
from itertools import chain


class EMGRecorder(Process):

    def __init__(self, path: os.PathLike or str, save_every: int = 10):
        super().__init__(daemon=True)
        self._base_path = os.path.join(path, "EMG_data")
        os.makedirs(self._base_path, exist_ok=True)
        self._start_rec_event: Event = Event()
        self._stop_rec_event: Event = Event()
        self._ready: Event = Event()
        self._save_every = save_every

    def _create_paths(self, sensors: list[str], channels: list[str]) -> list[os.PathLike or str]:
        paths, chans = [], []
        for sensor, channels in zip(sensors, channels):
            parent_path = os.path.join(self._base_path, sensor)
            os.mkdir(parent_path)
            path = os.path.join(parent_path, "data.h5")
            paths.append(path)
            with h5py.File(path, "w") as file:
                for chan in channels:
                    file.create_dataset(chan,
                                        shape=(0, 1),
                                        maxshape=(None, 1),
                                        dtype="float32",
                                        chunks=True)
        return paths

    @staticmethod
    def _save_data(
            data: list[list[float]], sensors: list[str], channels: list[list[str]], paths: list[os.PathLike or str]):
        if not all(data):
            return
        data = iter([np.array(x) for x in data])
        for sensor, sen_channels, path in zip(sensors, channels, paths):
            for channel in sen_channels:
                with h5py.File(path, "a") as file:
                    dset = file[channel]
                    bundle = np.atleast_2d(next(data)).T
                    dset.resize((dset.shape[0] + bundle.shape[0]), axis=0)
                    dset[-bundle.shape[0]:, :] = bundle

    def run(self):
        with DelsysCentro() as dcentro:
            if dcentro is None:
                return
            dcentro.scan()
            dcentro.start_station()
            sensors, channels = dcentro.sensors, dcentro.channels
            paths = self._create_paths(sensors, channels)
            dcentro.save_sensor_setup(self._base_path)
            i = 0
            data = [[] for _ in list(chain.from_iterable(channels))]
            while True:
                data_bundles = dcentro.get_data()
                if data_bundles is None:
                    continue
                if not self._ready.is_set() and dcentro.is_ready:
                    self._ready.set()
                if self._start_rec_event.is_set():
                    # todo: save only part of the data_bundle
                    data = [prev_data + data_bundle for prev_data, data_bundle in zip(data, data_bundles)]
                    i += 1
                if i > self._save_every:
                    self._save_data(data, sensors, channels, paths)
                    data = [d.clear() or d for d in data]
                    i = 0
                if self._stop_rec_event.is_set():
                    # todo: save data only partly
                    self._save_data(data, sensors, channels, paths)
                    break

    @property
    def is_ready(self):
        return self._ready.is_set()

    def start_recording(self):
        if self.is_ready:
            self._start_rec_event.set()

    def stop_recording(self):
        self._stop_rec_event.set()

if __name__ == "__main__":
    from data_rec import _create_folder
    er = EMGRecorder(_create_folder())
    er.start()
    while True:
        if er.is_ready:
            break
    er.start_recording()
    time.sleep(5)
    print("stop recording")
    er.stop_recording()
    time.sleep(5)



