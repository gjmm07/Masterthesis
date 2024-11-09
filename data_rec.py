from collections import deque
import cv2
from pynput import keyboard
import asyncio
import numpy as np
import os
from datetime import datetime
from OrthosisMotorController import FaulhaberMotorController, ELBOW_MIN_POSITION, FOREARM_MIN_POSITION
from ViconMoCap import ViconMoCap

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

class DataRecorderManager:

    def __init__(self):
        self._path = _create_folder()
        self._cur_pos: deque[tuple[float, float]] = deque(maxlen=1)
        self._stop_event: asyncio.Event = asyncio.Event()
        self._write_data: asyncio.Event = asyncio.Event()
        if not (has_mm_human3d or has_vicon):
            raise ValueError("Needs either vicon or mmhumand3d")
        if has_mm_human3d:
            self._hum_pose_est = HumanPoseEstimator(self._cur_pos, self._stop_event, cam_id=0)
        if has_vicon:
            self._hum_pose_est = ViconMoCap(self._cur_pos, self._stop_event)
        if not (has_delsys or has_digital_trigger):
            raise ValueError("Either needs digital trigger or Delsys to connect EMG")
        if has_digital_trigger:
            self._emg_recorder = Trigger()
        if has_delsys:
            self._emg_recorder = EMGRecorder(self._path)
            self._emg_recorder.start()
        self._pipeline = iter((self._sync_ort, self._start_rec_data, self._stop_rec_data, None))
        self._next_state = next(self._pipeline)
        self._include_ort = False
        if self._include_ort:
            self._ort_controller = FaulhaberMotorController("/dev/ttyUSB0")

    async def main(self):
        tasks = [
            self._hum_pose_est,
            self._keyboard_input(),
            # self.save_data()
        ]
        if self._include_ort:
            que = asyncio.Queue()
            tasks += [self._ort_controller.connect_device(),
                      self._ort_controller.start_background_tasks(que),
                      self._ort_controller.home()]
        await asyncio.gather(*tasks)

    async def _keyboard_input(self):
        loop = asyncio.get_event_loop()
        key_queue = transmit_keys(loop)
        while True:
            key = await key_queue.get()
            if key == keyboard.Key.esc:
                self._stop_event.set()
                break
            elif key == keyboard.Key.enter:
                if self._next_state is None:
                    break
                self._next_state()
        await self.__aexit__(None, None, None)

    def _sync_ort(self):
        # if not self._ort_controller.is_connected:
        #     return

        async def sync_ort():
            i = 0
            while not self._stop_event.is_set():
                await asyncio.sleep(1)
                print(self._cur_pos.popleft())
                # frame, full_pose = self._cur_pos.popleft()
                # cv2.imwrite(f"img_{i}.jpg", frame)
                # with open(f"pose_{i}.npz", "wb") as pose_f:
                #     np.save(pose_f, full_pose)
                # print(self._cur_pos.popleft())
                i += 1
        print("sync ort")
        self._next_state = next(self._pipeline)
        asyncio.ensure_future(sync_ort())

    def _start_rec_data(self):
        print("start rec data")
        self._emg_recorder.start_recording()
        self._next_state = next(self._pipeline)

    def _stop_rec_data(self):
        print("stop rec data")
        self._emg_recorder.stop_recording()
        self._next_state = next(self._pipeline)

    async def __aenter__(self):
        print("enter")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._include_ort:
            await self._ort_controller.stop()
        self._stop_event.set()
        self._write_data.set()
        self._hum_pose_est.stop()
        print("exit")


async def main():
    async with DataRecorderManager() as dr:
        await dr.main()


if __name__ == '__main__':
    asyncio.run(main())