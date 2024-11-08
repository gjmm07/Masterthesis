import asyncio
import os
import time
from multiprocessing import Process, Event
from DelsysCentro import DelsysCentro


class EMGRecorder(Process):

    def __init__(self, path: os.PathLike or str, start_rec_event: asyncio.Event, stop_rec_event: asyncio.Event):
        super().__init__(daemon=True)
        self._path = os.path.join(path, "EMG_data")
        os.makedirs(self._path)
        self._start_rec_event: Event = Event()
        self._stop_rec_event: Event = Event()

    def run(self):
        with DelsysCentro() as dcentro:
            while True:
                dcentro.get_data()
                if self._start_rec_event.is_set():
                    pass
                if self._stop_rec_event.is_set():
                    break

    def start_recording(self):
        self._start_rec_event.set()

    def stop_recording(self):
        self._stop_rec_event.set()



