from vicon_dssdk import ViconDataStream
from collections import deque
import asyncio


class ViconMoCap:

    def __init__(self, cur_pos: deque[tuple[float, float]], stop_event: asyncio.Event):
        self._cur_pos = cur_pos
        self._stop_event = stop_event

    def __await__(self):
        print("hi")
        self._cur_pos.append((0.0, 1.0))
        if not self._stop_event.is_set():
            yield from asyncio.sleep(0.5).__await__()
            yield from self.__await__()

    def stop(self):
        pass


