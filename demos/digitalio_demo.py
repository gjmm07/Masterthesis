import board
import digitalio
import time


led_green = digitalio.DigitalInOut(board.C0)
led_green.direction = digitalio.Direction.OUTPUT

# led_red = digitalio.DigitalInOut(board.C1)
# led_red.direction = digitalio.Direction.OUTPUT
led_green.value = False
# led_red.value = False

led_green.value = True
time.sleep(0.1)
led_green.value = False
time.sleep(5)
led_green.value = True
time.sleep(0.1)
led_green.value = False
# led_red.value = True
# time.sleep(1)
# led_red.value = False



