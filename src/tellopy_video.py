from __future__ import annotations

import threading
import time
from typing import Optional

import av
import numpy as np

from tellopy_core import TelloController


class TelloVideoStream:
    """Decode tellopy H264 stream via PyAV in a background thread."""

    def __init__(self, controller: TelloController) -> None:
        self.controller = controller
        self._container: Optional[av.container.InputContainer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self.last_error: str = ""

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.last_error = ""
        self._thread = threading.Thread(target=self._decode_loop, name="TelloVideoDecode", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._latest_rgb = None
        self._close_container()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_rgb is None:
                return None
            return self._latest_rgb.copy()

    def _decode_loop(self) -> None:
        frame_skip = 60
        while not self._stop_event.is_set():
            try:
                self._container = av.open(self.controller.drone.get_video_stream())
                for frame in self._container.decode(video=0):
                    if self._stop_event.is_set():
                        break
                    if frame_skip > 0:
                        frame_skip -= 1
                        continue

                    start = time.time()
                    rgb = np.array(frame.to_image())
                    with self._lock:
                        self._latest_rgb = rgb

                    time_base = max(float(frame.time_base), 1.0 / 60.0)
                    frame_skip = int((time.time() - start) / time_base)
            except Exception as exc:
                self.last_error = str(exc)
                if self._stop_event.is_set():
                    break
                # Startup packets may be undecodable; retry opening stream.
                time.sleep(0.3)
            finally:
                self._close_container()

    def _close_container(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
