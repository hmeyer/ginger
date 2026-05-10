"""Hold pan/tilt servos at 90° for mechanical assembly.

Run before attaching rocker arms to servos.
Press Ctrl-C when done installing.
"""

import signal
import sys
import time

from ginger.hardware.pca9685 import PCA9685
from ginger.hardware.servo import PanTilt


def main() -> None:
    print("Holding pan and tilt servos at 90° (center position).")
    print("Install the rocker arms and pan-tilt hardware now.")
    print("Press Ctrl-C when done.\n")

    pwm = PCA9685()
    pwm.set_pwm_freq(50)
    pan_tilt = PanTilt(pwm)

    def shutdown(sig, frame):
        print("\nServos released. Assembly complete.")
        pwm.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    pan_tilt.center()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
