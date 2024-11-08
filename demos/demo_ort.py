from OrthosisMotorController import FaulhaberMotorController
import asyncio
from OrthosisMotorController.Motions import Motions

async def disconnect_after():
    await asyncio.sleep(20)
    await mc.stop()

# port = "/dev/ttyUSB0"
port = "COM5"
mc = FaulhaberMotorController(port)
que: asyncio.Queue[Motions] = asyncio.Queue()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(asyncio.gather(
    mc.connect_device(),
    mc.home(),
    mc.start_background_tasks(que),
    disconnect_after()))