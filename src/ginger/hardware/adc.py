"""ADS7830 8-channel 8-bit ADC over I2C (address 0x48).

PCB V2.0: voltage coefficient 5.2, battery multiplier 2.
"""

import smbus2

_ADDRESS = 0x48
_COMMAND = 0x84  # single-ended, internal ref + ADC on

# Maps logical channel number to the ADS7830 MUX bits
_CHANNEL_MAP = {i: ((i << 2) | (i >> 1)) & 0x07 for i in range(8)}

_VOLTAGE_COEFF_V1 = 3.3
_VOLTAGE_COEFF_V2 = 5.2
_BATTERY_MULT_V1 = 3
_BATTERY_MULT_V2 = 2


class ADC:
    def __init__(self, pcb_version: int = 2, bus: int = 1):
        self._bus = smbus2.SMBus(bus)
        self._v_coeff = _VOLTAGE_COEFF_V2 if pcb_version == 2 else _VOLTAGE_COEFF_V1
        self._batt_mult = _BATTERY_MULT_V2 if pcb_version == 2 else _BATTERY_MULT_V1

    def read_raw(self, channel: int) -> int:
        """Return raw 8-bit ADC value for channel 0–7."""
        cmd = _COMMAND | (_CHANNEL_MAP[channel] << 4)
        self._bus.write_byte(_ADDRESS, cmd)
        # Read until two consecutive reads agree (noise filter)
        while True:
            v1 = self._bus.read_byte(_ADDRESS)
            v2 = self._bus.read_byte(_ADDRESS)
            if v1 == v2:
                return v1

    def read_voltage(self, channel: int) -> float:
        """Return voltage (V) for the given channel."""
        return round(self.read_raw(channel) / 255.0 * self._v_coeff, 2)

    def read_battery(self) -> float:
        """Return battery voltage (V). Wired to channel 2 with a voltage divider."""
        return round(self.read_voltage(2) * self._batt_mult, 2)

    def close(self) -> None:
        self._bus.close()
