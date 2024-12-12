import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from DelsysCentro import DelsysCentro
import threading
from itertools import chain
import numpy as np


class EMGPlotter:
    n = 9000
    # todo:
    #  a) Depending on sample rate the plot should be spaced
    #  b) Skin Check channel should not be offset
    #  c) maybe downsample the plotting data -> 9000 Samples lots

    def __init__(self):
        self._stop_event = threading.Event()
        self._data_queues: list[list[deque[float]]] = []
        self._dcentro: DelsysCentro or None = None
        self._lines = []
        lock = threading.Lock()
        thd = threading.Thread(target=self._setup_delsys, args=(lock, ))
        thd.start()
        self._plot(lock)

    def _plot(self, lock: threading.Lock):
        with lock:
            fig, axs = plt.subplots(len(self._dcentro.sensors), squeeze=False)
            offsets = []
            for ax, chans in zip(axs.flatten(), self._dcentro.channels):
                ax.set_xlim(0, EMGPlotter.n)
                for i, chan in enumerate(chans):
                    self._lines.append(*ax.plot([], [], linewidth=0.5, label=chan))
                    offsets.append(i)
                ax.set_ylim(-2, 2 + i)
                ax.legend(fontsize=10, loc="upper right")
            ani = FuncAnimation(fig, func=self._ani, fargs=(offsets, ), interval=100, cache_frame_data=False)
            plt.show()
            self._stop_event.set()

    def _ani(self, _, offsets):
        if not self._data_queues:
            return self._lines
        for line, que, oset in zip(self._lines, self._data_queues, offsets):
            plot_data = np.array(que)
            line.set_data(np.linspace(0, len(plot_data), len(plot_data)), np.array(plot_data) + oset)
        return self._lines

    def _setup_delsys(self, lock: threading.Lock):
        lock.acquire()
        with DelsysCentro() as self._dcentro:
            if self._dcentro is None:
                return
            self._dcentro.scan()
            self._dcentro.start_station()
            sensors, channels = self._dcentro.sensors, self._dcentro.channels
            self._data_queues = [deque(maxlen=EMGPlotter.n) for chans in channels for chan in chans]
            lock.release()
            self._get_data()

    def _get_data(self):
        while not self._stop_event.is_set():
            data_bundles = self._dcentro.get_data()
            if data_bundles is None:
                continue
            for que, bundle in zip(self._data_queues, data_bundles):
                que += deque(bundle)


if __name__ == "__main__":
    EMGPlotter()






