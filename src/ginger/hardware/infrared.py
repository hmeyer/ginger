"""3-sensor infrared line tracker.

Sensors are active-low: value=1 means line detected (dark surface).
GPIO pins: left=14, center=15, right=23.
"""

from gpiozero import LineSensor

_PINS = {"left": 14, "center": 15, "right": 23}


class InfraredSensors:
    def __init__(self):
        self._sensors = {name: LineSensor(pin) for name, pin in _PINS.items()}

    def read(self, sensor: str) -> bool:
        """True = line detected under this sensor."""
        return bool(self._sensors[sensor].value)

    def read_all(self) -> dict[str, bool]:
        return {name: bool(s.value) for name, s in self._sensors.items()}

    def close(self) -> None:
        for s in self._sensors.values():
            s.close()
