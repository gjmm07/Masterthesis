from OrthosisMotorController.FaulhaberMotorController import FaulhaberMotorController
import asyncio
from OrthosisMotorController.Motions import Motions

async def disconnect_after():
    await asyncio.sleep(20)
    await mc.stop()


mc = FaulhaberMotorController("/dev/ttyUSB0")
que: asyncio.Queue[Motions] = asyncio.Queue()
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(
    mc.connect_device(),
    mc.home(),
    mc.start_background_tasks(que),
    disconnect_after()))