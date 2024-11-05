from collections import deque

import cv2
from pynput import keyboard
import asyncio
from OrthosisMotorController import FaulhaberMotorController, ELBOW_MIN_POSITION, FOREARM_MIN_POSITION
from EMGTrigger import Trigger
import numpy as np


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
        from PoseEstimator import HumanPoseEstimator
        self._stop_event: asyncio.Event = asyncio.Event()
        self._ort_controller = FaulhaberMotorController("/dev/ttyUSB0")
        self._cur_pos: deque[tuple[float, float]] = deque(maxlen=1)
        self._hum_pose_est = HumanPoseEstimator(self._cur_pos, self._stop_event, cam_id=0)
        self._pipeline = iter((self._sync_ort, self._rec_data, self._rec_data, None))
        self._next_state = next(self._pipeline)
        self._write_data: asyncio.Event = asyncio.Event()
        self._emg_trigger = Trigger()

        self._include_ort = False

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
        self._stop_event.set()
        await self._ort_controller.stop()
        self._hum_pose_est.stop()
        self._write_data.set()

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

    def _rec_data(self):
        print("rec data")
        self._write_data.set()
        self._next_state = next(self._pipeline)

    async def save_data(self):
        await self._write_data.wait()
        if self._stop_event.is_set():
            return
        self._write_data.clear()
        await self._emg_trigger.trigger()
        while not self._stop_event.is_set() and not self._write_data.is_set():
            print("saving")
            await asyncio.sleep(1)
        await self._emg_trigger.trigger()

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