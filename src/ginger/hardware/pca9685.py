"""PCA9685 16-channel 12-bit PWM driver over I2C."""

import math
import time
import smbus2


_MODE1 = 0x00
_PRESCALE = 0xFE
_LED0_ON_L = 0x06

_OSC_FREQ = 25_000_000  # 25 MHz internal oscillator
_PWM_RESOLUTION = 4096  # 12-bit


class PCA9685:
    def __init__(self, address: int = 0x40, bus: int = 1):
        self._bus = smbus2.SMBus(bus)
        self._address = address
        self._write(_MODE1, 0x00)

    def set_pwm_freq(self, freq: float) -> None:
        prescale = round(_OSC_FREQ / (_PWM_RESOLUTION * freq)) - 1
        old_mode = self._read(_MODE1)
        self._write(_MODE1, (old_mode & 0x7F) | 0x10)  # sleep
        self._write(_PRESCALE, prescale)
        self._write(_MODE1, old_mode)
        time.sleep(0.005)
        self._write(_MODE1, old_mode | 0x80)  # restart

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        base = _LED0_ON_L + 4 * channel
        self._bus.write_i2c_block_data(
            self._address, base,
            [on & 0xFF, on >> 8, off & 0xFF, off >> 8],
        )

    def set_duty(self, channel: int, duty: int) -> None:
        """duty in range [-4095, 4095]; 0 = brake (both high)."""
        if duty == 0:
            self.set_pwm(channel, 0, _PWM_RESOLUTION - 1)
        elif duty > 0:
            self.set_pwm(channel, 0, min(duty, _PWM_RESOLUTION - 1))
        else:
            self.set_pwm(channel, 0, min(-duty, _PWM_RESOLUTION - 1))

    def set_servo_pulse_us(self, channel: int, pulse_us: float) -> None:
        """pulse_us: pulse width in microseconds (500–2500 for standard servos at 50 Hz)."""
        off = int(pulse_us * _PWM_RESOLUTION / 20_000)
        self.set_pwm(channel, 0, off)

    def close(self) -> None:
        self._bus.close()

    def _write(self, reg: int, value: int) -> None:
        self._bus.write_byte_data(self._address, reg, value)

    def _read(self, reg: int) -> int:
        return self._bus.read_byte_data(self._address, reg)
