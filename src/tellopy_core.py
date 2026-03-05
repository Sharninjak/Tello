from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import tellopy


@dataclass
class TelloState:
    connected: bool = False
    flying: bool = False
    battery: int = -1
    height_cm: int = -1
    wifi: int = -1


class TelloController:
    """Thread-safe wrapper around tellopy control and telemetry APIs."""

    def __init__(self) -> None:
        self.drone = tellopy.Tello()
        self.state = TelloState()
        self._lock = threading.Lock()
        self._connected = False
        self._axis = {"fb": 0, "lr": 0, "ud": 0, "yaw": 0}
        self._event_bound = False

    def connect(self, timeout: float = 30.0) -> None:
        with self._lock:
            if self._connected:
                return
            self.drone.connect()
            self.drone.wait_for_connection(timeout)
            if not self._event_bound:
                self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self._on_flight_data)
                self._event_bound = True
            self._connected = True
            self.state.connected = True

    def disconnect(self) -> None:
        with self._lock:
            if not self._connected:
                return
            self._stop_all_locked()
            self._connected = False
            self.state.connected = False
        try:
            self.drone.quit()
        except Exception:
            pass

    def takeoff(self) -> None:
        with self._lock:
            self._require_connected_locked()
            self.drone.takeoff()
            self.state.flying = True

    def land(self) -> None:
        with self._lock:
            if not self._connected:
                return
            self.drone.land()
            self._stop_all_locked()
            self.state.flying = False

    def emergency(self) -> None:
        with self._lock:
            if not self._connected:
                return
            self.drone.send_packet_data(b"emergency")
            self._stop_all_locked()
            self.state.flying = False

    def update_axis(self, *, fb: int, lr: int, ud: int, yaw: int) -> None:
        with self._lock:
            if not self._connected:
                return
            self._axis["fb"] = int(fb)
            self._axis["lr"] = int(lr)
            self._axis["ud"] = int(ud)
            self._axis["yaw"] = int(yaw)
            self._dispatch_axis_locked()

    def hover(self) -> None:
        with self._lock:
            if not self._connected:
                return
            self._axis = {"fb": 0, "lr": 0, "ud": 0, "yaw": 0}
            self._dispatch_axis_locked()

    def get_state_snapshot(self) -> TelloState:
        with self._lock:
            return TelloState(
                connected=self.state.connected,
                flying=self.state.flying,
                battery=self.state.battery,
                height_cm=self.state.height_cm,
                wifi=self.state.wifi,
            )

    def _dispatch_axis_locked(self) -> None:
        speed_fb = max(-100, min(100, self._axis["fb"]))
        speed_lr = max(-100, min(100, self._axis["lr"]))
        speed_ud = max(-100, min(100, self._axis["ud"]))
        speed_yaw = max(-100, min(100, self._axis["yaw"]))

        if speed_fb > 0:
            self.drone.forward(speed_fb)
            self.drone.backward(0)
        elif speed_fb < 0:
            self.drone.forward(0)
            self.drone.backward(-speed_fb)
        else:
            self.drone.forward(0)
            self.drone.backward(0)

        if speed_lr > 0:
            self.drone.right(speed_lr)
            self.drone.left(0)
        elif speed_lr < 0:
            self.drone.right(0)
            self.drone.left(-speed_lr)
        else:
            self.drone.right(0)
            self.drone.left(0)

        if speed_ud > 0:
            self.drone.up(speed_ud)
            self.drone.down(0)
        elif speed_ud < 0:
            self.drone.up(0)
            self.drone.down(-speed_ud)
        else:
            self.drone.up(0)
            self.drone.down(0)

        if speed_yaw > 0:
            self.drone.clockwise(speed_yaw)
            self.drone.counter_clockwise(0)
        elif speed_yaw < 0:
            self.drone.clockwise(0)
            self.drone.counter_clockwise(-speed_yaw)
        else:
            self.drone.clockwise(0)
            self.drone.counter_clockwise(0)

    def _stop_all_locked(self) -> None:
        self._axis = {"fb": 0, "lr": 0, "ud": 0, "yaw": 0}
        try:
            self._dispatch_axis_locked()
        except Exception:
            pass

    def _require_connected_locked(self) -> None:
        if not self._connected:
            raise RuntimeError("Tello is not connected")

    def _on_flight_data(
        self,
        event: Any = None,
        sender: Any = None,
        data: Any = None,
        **_kwargs: Any,
    ) -> None:
        del event, sender
        if data is None:
            return
        # tellopy flight-data fields vary by firmware; try common names.
        self.state.battery = _coalesce_int(data, ["battery_percentage", "battery_low", "battery"])
        self.state.height_cm = _coalesce_int(data, ["height", "height_cm", "tof"])
        self.state.wifi = _coalesce_int(data, ["wifi_strength", "wifi"])


def _coalesce_int(obj: Any, names: list[str]) -> int:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if isinstance(value, bool):
                continue
            try:
                return int(value)
            except Exception:
                continue
    return -1
