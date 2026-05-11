"""8× WS2812B RGB LED strip via SPI (PCB V2.0, GRB order, SPI bus 0).

Uses MOSI (GPIO 10) at ~6.4 MHz to bit-bang WS2812B timing.
"""

import numpy as np
import spidev

_LED_COUNT = 8
_SPI_BUS = 0
_SPI_DEVICE = 0
_SPI_HZ = int(8 / 1.25e-6)  # 6.4 MHz


class LEDStrip:
    def __init__(self, count: int = _LED_COUNT, brightness: int = 255):
        self._count = count
        self._brightness = brightness
        # GRB order offsets
        self._g, self._r, self._b = 0, 1, 2
        self._buf = [[0, 0, 0]] * count  # stored as [G, R, B]

        self._spi = spidev.SpiDev()
        self._spi.open(_SPI_BUS, _SPI_DEVICE)
        self._spi.mode = 0
        self.clear()

    def set(self, index: int, r: int, g: int, b: int) -> None:
        scale = self._brightness / 255
        self._buf[index] = [
            round(g * scale),
            round(r * scale),
            round(b * scale),
        ]

    def set_all(self, r: int, g: int, b: int) -> None:
        for i in range(self._count):
            self.set(i, r, g, b)

    def show(self) -> None:
        flat = np.array([v for pixel in self._buf for v in pixel], dtype=np.uint8)
        tx = np.zeros(len(flat) * 8, dtype=np.uint8)
        for bit in range(8):
            tx[7 - bit::8] = ((flat >> bit) & 1) * 0x78 + 0x80
        self._spi.xfer(tx.tolist(), _SPI_HZ)

    def clear(self) -> None:
        self.set_all(0, 0, 0)
        self.show()

    def close(self) -> None:
        self.clear()
        self._spi.close()
