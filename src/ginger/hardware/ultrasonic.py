"""HC-SR04 ultrasonic distance sensor via gpiozero.

Trigger: GPIO 27, Echo: GPIO 22. Max range: 3 m.
"""

import warnings
from gpiozero import DistanceSensor
from gpiozero.exc import DistanceSensorNoEcho, PWMSoftwareFallback

_TRIGGER = 27
_ECHO = 22
_MAX_DISTANCE_M = 3.0


class Ultrasonic:
    def __init__(self):
        warnings.filterwarnings("ignore", category=DistanceSensorNoEcho)
        warnings.filterwarnings("ignore", category=PWMSoftwareFallback)
        self._sensor = DistanceSensor(
            echo=_ECHO, trigger=_TRIGGER, max_distance=_MAX_DISTANCE_M
        )

    def distance_cm(self) -> float | None:
        """Return distance in cm, or None if out of range."""
        try:
            return round(self._sensor.distance * 100, 1)
        except RuntimeWarning:
            return None

    def close(self) -> None:
        self._sensor.close()
