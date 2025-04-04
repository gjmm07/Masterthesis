from asyncio import gather
from collections import deque
from typing import Optional, Sequence
import cv2
from tqdm import tqdm
from pynput import keyboard
import asyncio
import numpy as np
import os
from datetime import datetime
from OrthosisMotorController import FaulhaberMotorController, ELBOW_MIN_POSITION, FOREARM_MIN_POSITION
from ViconMoCap import ViconMoCapSegment, ViconMoCapMarker

### Import EMG recording modules ( either has to work )
try:
    # Use this with delsys Trigno discover running waiting for digital trigger ( Using Adafruit FT232H )
    from EMGTrigger import Trigger
    has_digital_trigger = True
except ModuleNotFoundError:
    has_digital_trigger = False

try:
    # Assumes delsys Centro is connected via USB ( only working on Windows )
    import DelsysCentro
    from EMGRecorder import EMGRecorder
    has_delsys = True
except ModuleNotFoundError:
    has_delsys = False

## Import Motion Capturing ( Either has to work )
try:
    # Open Source motion capturing using mmhuman3d (https://github.com/open-mmlab/mmhuman3d) Only working on Linux
    from PoseEstimator import HumanPoseEstimator
    has_mm_human3d = True
except ModuleNotFoundError:
    has_mm_human3d = False

try:
    # prophysics vicon motion Capturing ( Vicon Nexus with appropriated labeled markers needs to be running )
    from vicon_dssdk import ViconDataStream
    has_vicon = True
except ModuleNotFoundError:
    has_vicon = False


from const import *


def _create_folder():
    """
    Creates a new folder inside recordings, where all submodules can place data
    :return: Empty folder
    """
    path = os.path.join("recordings", datetime.now().strftime("%d-%m-%y--%H-%M-%S"))
    os.makedirs(path)
    return path

def transmit_keys(loop: asyncio.AbstractEventLoop):
    queue = asyncio.Queue()
    def on_release(key):
        try:
            key = key.char
        except AttributeError:
            pass
        loop.call_soon_threadsafe(queue.put_nowait, key)
    keyboard.Listener(on_release=on_release, suppress=True).start()
    return queue

class _DataRecorder:

    def __init__(self, stop_event: asyncio.Event, pipeline: Sequence):
        self._stop_event: asyncio.Event = stop_event
        self._pipeline = iter(pipeline)
        self._next_state = next(self._pipeline)
        self._path = _create_folder()
        self._save_name()

    def _save_name(self):
        with open(os.path.join(self._path, "info.txt"), "w") as file:
            file.write(f"{SUBJECT}\n")

    async def _keyboard_input(self):
        loop = asyncio.get_event_loop()
        key_queue = transmit_keys(loop)
        while True:
            key = await key_queue.get()
            if key == keyboard.Key.esc:
                print("esc")
                self._stop_event.set()
                break
            elif key == keyboard.Key.enter:
                if self._next_state is None:
                    break
                self._next_state()
        await self.__aexit__(None, None, None)

    async def __aenter__(self):
        raise NotImplementedError("Please implement in inheriting class")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError("Please implement in inheriting class")


class InitMoCap(_DataRecorder):

    def __init__(self):
        self._start_rec = asyncio.Event()
        self._stop_rec: asyncio.Event = asyncio.Event()
        self._stop_event: asyncio.Event = asyncio.Event()
        super().__init__(self._stop_event, (self._start_rec_data, self._stop_rec_data, None))
        self._hum_pose_est = ViconMoCapMarker.collect_only(
            self._stop_event, self._path, self._start_rec, self._stop_rec
        )
        self._append_info()

    def _append_info(self):
        with open(os.path.join(self._path, "info.txt"), "a") as file:
            file.write("init mocap\n")

    async def main(self):
        tasks = (self._hum_pose_est.run(),
                 self._keyboard_input())
        await asyncio.gather(*tasks)

    async def _save_model(self):
        await asyncio.sleep(0.3) # give time to end data writing
        self._hum_pose_est.save_marker_model()

    def _start_rec_data(self):
        print("start rec")
        self._start_rec.set()
        self._next_state = next(self._pipeline)

    def _stop_rec_data(self):
        print("stop rec")
        self._stop_rec.set()
        asyncio.ensure_future(self._save_model())
        self._next_state = next(self._pipeline)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._start_rec.set()
        self._stop_rec.set()
        self._stop_event.set()

class DataRecorderManager(_DataRecorder):

    def __init__(self):
        self._stop_event: asyncio.Event = asyncio.Event()
        pipeline = []
        if INCLUDE_MOCAP:
            pipeline.append(self._set_zero)
        if INCLUDE_ORT and INCLUDE_MOCAP:
            pipeline.append(self._sync_ort)
        pipeline += [self._start_rec_data, self._stop_rec_data, None]
        super().__init__(self._stop_event,
                         pipeline)
        self._cur_pos: deque[tuple[float, float]] = deque(maxlen=1)
        self._start_rec: asyncio.Event = asyncio.Event()
        self._stop_rec: asyncio.Event = asyncio.Event()
        if not (has_mm_human3d or has_vicon):
            raise ValueError("Needs either vicon or mmhumand3d")
        if has_mm_human3d and INCLUDE_MOCAP:
            self._hum_pose_est = HumanPoseEstimator(self._cur_pos, self._stop_event, cam_id=0)
        if has_vicon and INCLUDE_MOCAP:
            self._hum_pose_est = ViconMoCapMarker(
                self._cur_pos, self._stop_event, self._path, self._start_rec, self._stop_rec)
        if not (has_delsys or has_digital_trigger):
            raise ValueError("Either needs digital trigger or Delsys to connect EMG")
        if has_digital_trigger:
            self._emg_recorder = Trigger()
        if has_delsys:
            self._emg_recorder = EMGRecorder(self._path)
            if INCLUDE_EMG:
                self._emg_recorder.start()
        if INCLUDE_ORT:
            self._ort_controller = FaulhaberMotorController(
                "COM5", save_path=self._path, start_rec=self._start_rec, stop_rec=self._stop_rec)
            self._ort_queue: asyncio.Queue[tuple[float | None, ...]] = asyncio.Queue(maxsize=1)

    async def main(self):
        if not any((INCLUDE_ORT, INCLUDE_EMG, INCLUDE_MOCAP)):
            return
        tasks = [
            self._keyboard_input(),
        ]
        if INCLUDE_MOCAP:
            tasks.append(self._hum_pose_est.run())
        if INCLUDE_ORT:
            tasks += [self._ort_controller.connect_device(),
                      self._ort_controller.start_background_tasks(self._ort_queue),
                      self._ort_controller.home()]
        await asyncio.gather(*tasks)

    def _set_zero(self):
        asyncio.ensure_future(self._hum_pose_est.set_zero(2))
        self._next_state = next(self._pipeline)

    def _sync_ort(self):
        if INCLUDE_ORT and not self._ort_controller.is_ready:
            print("Orthosis not ready")
            return

        async def sync_ort():
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)
                if INCLUDE_ORT:
                    try:
                        self._ort_queue.put_nowait(self._cur_pos[-1])
                    except asyncio.QueueFull:
                        pass
                else:
                    print([round(x) for x in self._cur_pos[-1]])
        print("sync ort")
        self._next_state = next(self._pipeline)
        asyncio.ensure_future(sync_ort())

    def _delay(self, delay: int or None, func, *args):
        if delay is None:
            return
        async def wait():
            await asyncio.sleep(0.1)
            for i in tqdm(range(delay), leave=False):
                await asyncio.sleep(1)
                if self._stop_event.is_set():
                    break
            func(*args)
        asyncio.ensure_future(wait())

    def _start_rec_data(self):
        if not self._emg_recorder.is_ready and INCLUDE_EMG:
            print("not ready for recording")
            return
        print("start rec data")
        self._emg_recorder.start_recording()
        self._start_rec.set()
        self._next_state = next(self._pipeline)
        self._delay(REC_DELAY, self._stop_rec_data)

    def _stop_rec_data(self):
        print("stop rec data")
        self._emg_recorder.stop_recording()
        self._stop_rec.set()
        self._next_state = next(self._pipeline)

    async def __aenter__(self):
        print("enter")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if INCLUDE_ORT:
            await self._ort_controller.stop()
        self._start_rec.set()
        self._stop_rec.set()
        self._stop_event.set()
        if INCLUDE_MOCAP:
            self._hum_pose_est.stop()
        print("exit")


async def main():
    async with DataRecorderManager() as dr:
        await dr.main()


async def init_mocap():
    async with InitMoCap() as imocap:
        await imocap.main()


if __name__ == '__main__':
    # asyncio.run(init_mocap())
    asyncio.run(main())