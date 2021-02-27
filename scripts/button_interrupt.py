#!/usr/bin/env python

import RPi.GPIO as GPIO
import time
import os

# Pin Definitions:
led_pin_1 = 13
led_pin_2 = 12
led_pin_3 = 21
led_pin_4 = 23
led_pin_5 = 29
but_pin1 = 18
but_pin2 = 15
but_pin3 = 19

# blink LED 2 quickly 5 times when button pressed
def blink1(channel):
    print("Blink LED 1")
    try:
        os.system('docker start open-camera')
        print("打开摄像机 {}".format(time.asctime(time.localtime(time.time()))))
    finally:
        for i in range(3):
            GPIO.output(led_pin_1, GPIO.LOW)
            time.sleep(0.5)
            GPIO.output(led_pin_1, GPIO.HIGH)
            time.sleep(0.5)

def blink2(channel):
    print("Blink LED 2")
    try:
        os.system('docker stop open-camera')
        print("关闭摄像机 {}".format(time.asctime( time.localtime(time.time()))))
        GPIO.output(led_pin_4, GPIO.HIGH)
        GPIO.output(led_pin_5, GPIO.HIGH)
    finally:
        for i in range(3):
            GPIO.output(led_pin_2, GPIO.LOW)
            time.sleep(0.5)
            GPIO.output(led_pin_2, GPIO.HIGH)
            time.sleep(0.5)

def blink3(channel):
    print("Blink LED 3")
    print("关闭机器 {}".format(time.asctime( time.localtime(time.time()))))
    for i in range(3):
        GPIO.output(led_pin_3, GPIO.LOW)
        time.sleep(0.5)
        GPIO.output(led_pin_3, GPIO.HIGH)
        time.sleep(0.5)
    os.system('shutdown -h now')

def main():

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup([led_pin_1, led_pin_2, led_pin_3, led_pin_4, led_pin_5], GPIO.OUT)  # LED pins set as output
    GPIO.setup(but_pin1, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # button pin set as input
    GPIO.setup(but_pin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # button pin set as input
    GPIO.setup(but_pin3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    # Initial state for LEDs:
    GPIO.output(led_pin_1, GPIO.HIGH)
    GPIO.output(led_pin_2, GPIO.HIGH)
    GPIO.output(led_pin_3, GPIO.HIGH)
    GPIO.output(led_pin_4, GPIO.HIGH)
    GPIO.output(led_pin_5, GPIO.HIGH)

    GPIO.add_event_detect(but_pin1, GPIO.FALLING, callback=blink1, bouncetime=200)
    GPIO.add_event_detect(but_pin2, GPIO.FALLING, callback=blink2, bouncetime=200)
    GPIO.add_event_detect(but_pin3, GPIO.FALLING, callback=blink3, bouncetime=200)

    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            time.sleep(1)
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    while(True):
        try:
            main()
        except:
            main()
