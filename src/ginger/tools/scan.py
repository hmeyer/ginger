"""Hardware presence scan — prints status of every subsystem."""

import smbus2
import spidev


def _check_i2c(address: int, bus: int = 1) -> bool:
    try:
        b = smbus2.SMBus(bus)
        b.read_byte(address)
        b.close()
        return True
    except OSError:
        return False


def _check_spi(bus: int = 0, device: int = 0) -> bool:
    try:
        s = spidev.SpiDev()
        s.open(bus, device)
        s.close()
        return True
    except OSError:
        return False


def _check_camera() -> bool:
    try:
        from picamera2 import Picamera2
        cams = Picamera2.global_camera_info()
        return len(cams) > 0
    except Exception:
        return False


def _check_gpio_input(pin: int) -> bool:
    try:
        from gpiozero import InputDevice
        d = InputDevice(pin)
        d.close()
        return True
    except Exception:
        return False


def main() -> None:
    print("=== Ginger hardware scan ===\n")

    checks = [
        ("PCA9685  (motors + servos)", _check_i2c(0x40)),
        ("ADS7830  (ADC / light / battery)", _check_i2c(0x48)),
        ("WS2812B  (LED strip via SPI0)", _check_spi(0, 0)),
        ("Camera   (CSI / picamera2)", _check_camera()),
        ("IR left  (GPIO 14)", _check_gpio_input(14)),
        ("IR center(GPIO 15)", _check_gpio_input(15)),
        ("IR right (GPIO 23)", _check_gpio_input(23)),
        ("Ultrasonic echo (GPIO 22)", _check_gpio_input(22)),
        ("Ultrasonic trig (GPIO 27)", _check_gpio_input(27)),
    ]

    for label, ok in checks:
        status = "OK " if ok else "---"
        print(f"  [{status}]  {label}")

    print()
    ok_count = sum(1 for _, ok in checks if ok)
    print(f"{ok_count}/{len(checks)} subsystems detected.")


if __name__ == "__main__":
    main()
