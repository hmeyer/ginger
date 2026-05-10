"""Pan/tilt servo control via PCA9685."""

from .pca9685 import PCA9685

# PCA9685 channels for pan and tilt servos
_PAN_CHANNEL = 8
_TILT_CHANNEL = 9

# Pulse width at 50 Hz: 500 µs = 0°, 1500 µs = 90°, 2500 µs = 180°
_PULSE_MIN_US = 500
_PULSE_MAX_US = 2500
_PULSE_CENTER_US = 1500
_ANGLE_RANGE = 180.0


def _angle_to_pulse(angle: float, invert: bool = False, trim_us: float = 0) -> float:
    angle = max(0.0, min(180.0, angle))
    pulse = _PULSE_MIN_US + angle / _ANGLE_RANGE * (_PULSE_MAX_US - _PULSE_MIN_US)
    if invert:
        pulse = _PULSE_MIN_US + _PULSE_MAX_US - pulse
    return pulse + trim_us


class PanTilt:
    """Controls the two-axis camera pan/tilt head.

    Pan (horizontal) servo is wired inverted relative to tilt.
    trim_us values allow per-unit mechanical offset correction.
    """

    def __init__(self, pwm: PCA9685, pan_trim_us: float = 0, tilt_trim_us: float = 0):
        self._pwm = pwm
        self._pan_trim = pan_trim_us
        self._tilt_trim = tilt_trim_us

    def set_pan(self, angle: float) -> None:
        """angle: 0–180 degrees. 90 = center."""
        pulse = _angle_to_pulse(angle, invert=True, trim_us=self._pan_trim)
        self._pwm.set_servo_pulse_us(_PAN_CHANNEL, pulse)

    def set_tilt(self, angle: float) -> None:
        """angle: 0–180 degrees. 90 = center."""
        pulse = _angle_to_pulse(angle, invert=False, trim_us=self._tilt_trim)
        self._pwm.set_servo_pulse_us(_TILT_CHANNEL, pulse)

    def center(self) -> None:
        self.set_pan(90)
        self.set_tilt(90)
