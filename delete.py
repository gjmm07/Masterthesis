import asyncio
import pynput


def transmit_keys():
    # Start a keyboard listener that transmits keypresses into an
    # asyncio queue, and immediately return the queue to the caller.
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    def on_press(key):
        # this callback is invoked from another thread, so we can't
        # just queue.put_nowait(key.char), we have to go through
        # call_soon_threadsafe
        loop.call_soon_threadsafe(queue.put_nowait, key.char)
    pynput.keyboard.Listener(on_press=on_press).start()
    return queue

async def main():
    key_queue = transmit_keys()
    while True:
        key = await key_queue.get()
        print(key)

asyncio.run(main())