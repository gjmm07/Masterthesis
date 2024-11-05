import asyncio
import board
import digitalio


class Trigger(digitalio.DigitalInOut):

    def __init__(self, pin=board.C0):
        super().__init__(pin)
        self.direction = digitalio.Direction.OUTPUT

    async def trigger(self):
        self.value = True
        await asyncio.sleep(0.5)
        self.value = False
        #

if __name__ == '__main__':
    t = Trigger()
    asyncio.run(t.trigger())

