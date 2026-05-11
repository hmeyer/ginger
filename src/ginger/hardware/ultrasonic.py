"""HC-SR04 ultrasonic distance sensor.

Trigger: GPIO 27 (output), Echo: GPIO 22 (input). Max range: 3 m.
Uses RPi.GPIO directly — avoids the lgpio "GPIO busy" issue that
gpiozero's DistanceSensor can leave behind after unclean exits.
"""

import time
import RPi.GPIO as GPIO

_TRIGGER = 27
_ECHO = 22
_ECHO_START_TIMEOUT_S = 0.01   # 10ms to wait for echo to go HIGH
_ECHO_END_TIMEOUT_S   = 0.04   # 40ms max pulse (HC-SR04 outputs 38ms on no echo)


class Ultrasonic:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(_TRIGGER, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(_ECHO, GPIO.IN)

    def distance_cm(self) -> float | None:
        """Return distance in cm, or None on timeout."""
        # 10 µs trigger pulse
        GPIO.output(_TRIGGER, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(_TRIGGER, GPIO.LOW)

        # Wait for echo HIGH
        deadline = time.monotonic() + _ECHO_START_TIMEOUT_S
        while GPIO.input(_ECHO) == GPIO.LOW:
            if time.monotonic() > deadline:
                return None

        t_start = time.monotonic()

        # Wait for echo LOW
        deadline = time.monotonic() + _ECHO_END_TIMEOUT_S
        while GPIO.input(_ECHO) == GPIO.HIGH:
            if time.monotonic() > deadline:
                return None

        t_end = time.monotonic()

        # Speed of sound: 34300 cm/s; divide by 2 for round trip
        return round((t_end - t_start) * 34300 / 2, 1)

    def close(self) -> None:
        GPIO.cleanup([_TRIGGER, _ECHO])
