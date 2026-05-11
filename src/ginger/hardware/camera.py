"""Picamera2 wrapper for streaming and capture."""

import io
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from libcamera import Transform


class _StreamBuffer(io.BufferedIOBase):
    def __init__(self):
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


class Camera:
    def __init__(
        self,
        stream_size: tuple[int, int] = (640, 480),
        hflip: bool = False,
        vflip: bool = False,
    ):
        self._cam = Picamera2()
        transform = Transform(hflip=int(hflip), vflip=int(vflip))
        self._stream_cfg = self._cam.create_video_configuration(
            main={"size": stream_size}, transform=transform
        )
        self._output = _StreamBuffer()
        self._streaming = False

    def start_stream(self) -> None:
        if self._streaming:
            return
        self._cam.configure(self._stream_cfg)
        self._cam.start_recording(JpegEncoder(), FileOutput(self._output))
        self._streaming = True

    def stop_stream(self) -> None:
        if not self._streaming:
            return
        self._cam.stop_recording()
        self._streaming = False

    def get_frame(self) -> bytes:
        """Block until a new JPEG frame is available, then return it."""
        with self._output.condition:
            self._output.condition.wait()
            return self._output.frame

    def capture_jpeg(self) -> bytes:
        """Capture a single JPEG without starting continuous streaming."""
        data = io.BytesIO()
        self._cam.start()
        self._cam.capture_file(data, format="jpeg")
        self._cam.stop()
        return data.getvalue()

    def close(self) -> None:
        self.stop_stream()
        self._cam.close()
