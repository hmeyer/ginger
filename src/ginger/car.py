"""Top-level Car class — coordinates all hardware with built-in safety."""

import time

from ginger.hardware.pca9685 import PCA9685
from ginger.hardware.motor import Motors
from ginger.hardware.servo import PanTilt
from ginger.hardware.ultrasonic import Ultrasonic
from ginger.hardware.infrared import InfraredSensors
from ginger.hardware.adc import ADC
from ginger.hardware.led import LEDStrip

SAFE_DISTANCE_CM = 30.0   # stop if anything closer than this
PAN_SETTLE_S     = 0.3    # wait after centering pan before reading US


class Car:
    def __init__(self, pcb_version: int = 2):
        self._pwm = PCA9685()
        self._pwm.set_pwm_freq(50)
        self.motors   = Motors(self._pwm)
        self.pan_tilt = PanTilt(self._pwm)
        self.us       = Ultrasonic()
        self.ir       = InfraredSensors()
        self.adc      = ADC(pcb_version=pcb_version)
        self.leds     = LEDStrip()

        # Start with pan centered so US points forward
        self.pan_tilt.center()

    # ------------------------------------------------------------------
    # Safety

    def clear_ahead(self) -> tuple[bool, float | None]:
        """Center pan, wait for settle, read US. Returns (safe, distance_cm)."""
        self.pan_tilt.set_pan(90)
        time.sleep(PAN_SETTLE_S)
        dist = self.us.distance_cm()
        safe = dist is None or dist > SAFE_DISTANCE_CM
        return safe, dist

    # ------------------------------------------------------------------
    # Movement

    def drive(self, left: int, right: int, duration_s: float) -> bool:
        """
        Drive for up to duration_s seconds.
        Checks clearance before starting; stops early if obstacle detected.
        Returns True if completed without obstruction, False if stopped early.
        """
        # Only check forward clearance when driving forward
        if left > 0 and right > 0:
            safe, dist = self.clear_ahead()
            if not safe:
                print(f'STOP: obstacle at {dist:.1f}cm')
                return False

        self.motors.drive(left, right)
        deadline = time.monotonic() + duration_s
        try:
            while time.monotonic() < deadline:
                # Mid-drive check every 100ms when going forward
                if left > 0 and right > 0:
                    dist = self.us.distance_cm()
                    if dist is not None and dist < SAFE_DISTANCE_CM:
                        print(f'STOP: obstacle at {dist:.1f}cm')
                        return False
                time.sleep(0.1)
        finally:
            self.motors.stop()
        return True

    def forward(self, duty: int = 2000, duration_s: float = 0.5) -> bool:
        return self.drive(duty, duty, duration_s)

    def backward(self, duty: int = 2000, duration_s: float = 0.5) -> bool:
        return self.drive(-duty, -duty, duration_s)

    def turn_left(self, duty: int = 2000, duration_s: float = 0.5) -> bool:
        return self.drive(-duty, duty, duration_s)

    def turn_right(self, duty: int = 2000, duration_s: float = 0.5) -> bool:
        return self.drive(duty, -duty, duration_s)

    def stop(self) -> None:
        self.motors.stop()

    # ------------------------------------------------------------------
    # Sensors

    def battery_v(self) -> float:
        return self.adc.read_battery()

    # ------------------------------------------------------------------
    # Lifecycle

    def close(self) -> None:
        self.motors.stop()
        self.pan_tilt.center()
        self.leds.clear()
        self.us.close()
        self.ir.close()
        self.adc.close()
        self.leds.close()
        self._pwm.close()
