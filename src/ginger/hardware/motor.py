"""4WD drive motor control via PCA9685."""

from .pca9685 import PCA9685

# PCA9685 channel pairs per wheel: (forward_channel, reverse_channel)
_WHEEL_CHANNELS = {
    "left_front":  (1, 0),
    "left_rear":   (2, 3),
    "right_front": (6, 7),
    "right_rear":  (4, 5),
}

_MAX_DUTY = 4095


class Motors:
    def __init__(self, pwm: PCA9685):
        self._pwm = pwm

    def set_wheel(self, wheel: str, duty: int) -> None:
        """duty: -4095 (full reverse) to 4095 (full forward), 0 = brake."""
        fwd, rev = _WHEEL_CHANNELS[wheel]
        duty = max(-_MAX_DUTY, min(_MAX_DUTY, duty))
        if duty > 0:
            self._pwm.set_pwm(fwd, 0, 0)
            self._pwm.set_pwm(rev, 0, duty)
        elif duty < 0:
            self._pwm.set_pwm(rev, 0, 0)
            self._pwm.set_pwm(fwd, 0, -duty)
        else:
            self._pwm.set_pwm(fwd, 0, _MAX_DUTY)
            self._pwm.set_pwm(rev, 0, _MAX_DUTY)

    def drive(self, left: int, right: int) -> None:
        """Set all wheels: left side / right side duty (-4095..4095)."""
        self.set_wheel("left_front", left)
        self.set_wheel("left_rear", left)
        self.set_wheel("right_front", right)
        self.set_wheel("right_rear", right)

    def stop(self) -> None:
        self.drive(0, 0)

    def close(self) -> None:
        self.stop()
