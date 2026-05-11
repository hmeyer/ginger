"""A small light show for the 8-LED WS2812B strip."""

import math
import time

from ginger.hardware.led import LEDStrip


def wheel(pos: int) -> tuple[int, int, int]:
    """Color wheel: 0–255 maps smoothly through the full hue circle."""
    pos = pos % 256
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    if pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    pos -= 170
    return (pos * 3, 0, 255 - pos * 3)


def chase(strip: LEDStrip, duration: float = 3.0) -> None:
    """Single pixel racing around the ring."""
    end = time.time() + duration
    i = 0
    while time.time() < end:
        strip.set_all(0, 0, 0)
        r, g, b = wheel(i * 32)
        strip.set(i % 8, r, g, b)
        strip.show()
        time.sleep(0.07)
        i += 1


def rainbow_cycle(strip: LEDStrip, duration: float = 4.0) -> None:
    """Each LED a different hue, cycling together."""
    end = time.time() + duration
    offset = 0
    while time.time() < end:
        for i in range(8):
            r, g, b = wheel((i * 32 + offset) % 256)
            strip.set(i, r, g, b)
        strip.show()
        time.sleep(0.02)
        offset = (offset + 2) % 256


def breathe(strip: LEDStrip, color: tuple[int, int, int], duration: float = 3.0) -> None:
    """Whole strip fades in and out on a sine curve."""
    r0, g0, b0 = color
    end = time.time() + duration
    t = 0.0
    while time.time() < end:
        brightness = (math.sin(t) + 1) / 2
        strip.set_all(int(r0 * brightness), int(g0 * brightness), int(b0 * brightness))
        strip.show()
        time.sleep(0.02)
        t += 0.08


def sparkle(strip: LEDStrip, duration: float = 3.0) -> None:
    """Random pixels flash white."""
    import random
    end = time.time() + duration
    while time.time() < end:
        strip.set_all(0, 0, 0)
        for _ in range(3):
            strip.set(random.randrange(8), 255, 255, 255)
        strip.show()
        time.sleep(0.06)


def wipe(strip: LEDStrip, duration: float = 3.0) -> None:
    """Colours fill and clear one pixel at a time."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 200, 0)]
    end = time.time() + duration
    ci = 0
    while time.time() < end:
        r, g, b = colors[ci % len(colors)]
        for i in range(8):
            strip.set(i, r, g, b)
            strip.show()
            time.sleep(0.05)
        for i in range(8):
            strip.set(i, 0, 0, 0)
            strip.show()
            time.sleep(0.05)
        ci += 1


def main() -> None:
    strip = LEDStrip(brightness=80)
    try:
        print("chase")
        chase(strip)
        print("rainbow")
        rainbow_cycle(strip)
        print("breathe blue")
        breathe(strip, (0, 80, 255))
        print("sparkle")
        sparkle(strip)
        print("wipe")
        wipe(strip)
        print("rainbow finale")
        rainbow_cycle(strip, duration=5.0)
    finally:
        strip.clear()
        strip.close()


if __name__ == "__main__":
    main()
