from __future__ import annotations

import os
import sys
from datetime import datetime

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from tellopy_core import TelloController
from tellopy_video import TelloVideoStream


KEY_MAP = {
    QtCore.Qt.Key_W: ("fb", 1),
    QtCore.Qt.Key_S: ("fb", -1),
    QtCore.Qt.Key_Q: ("lr", -1),
    QtCore.Qt.Key_E: ("lr", 1),
    QtCore.Qt.Key_R: ("ud", 1),
    QtCore.Qt.Key_F: ("ud", -1),
    QtCore.Qt.Key_A: ("yaw", -1),
    QtCore.Qt.Key_D: ("yaw", 1),
}


class TelloControlPanel(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tello tellopy Control Panel")
        self.resize(1220, 760)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.controller = TelloController()
        self.video = TelloVideoStream(self.controller)
        self.pressed_keys: set[int] = set()
        self.speed = 35
        self.latest_frame = None

        self._build_ui()
        self._bind_signals()

        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self._render_video)
        self.video_timer.start(33)

        self.control_timer = QtCore.QTimer(self)
        self.control_timer.timeout.connect(self._control_tick)
        self.control_timer.start(50)

        self.telemetry_timer = QtCore.QTimer(self)
        self.telemetry_timer.timeout.connect(self._refresh_telemetry)
        self.telemetry_timer.start(200)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        self.video_label = QtWidgets.QLabel("Video Offline")
        self.video_label.setMinimumSize(860, 640)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#101417;color:#d9e2ec;border:1px solid #334e68;")

        side = QtWidgets.QFrame()
        side.setFixedWidth(320)
        side.setStyleSheet("background:#1f2933;color:#f0f4f8;")
        side_layout = QtWidgets.QVBoxLayout(side)

        title = QtWidgets.QLabel("Flight Control")
        title.setStyleSheet("font-size:20px;font-weight:700;")

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_takeoff = QtWidgets.QPushButton("Takeoff")
        self.btn_land = QtWidgets.QPushButton("Land")
        self.btn_emergency = QtWidgets.QPushButton("Emergency")
        self.btn_photo = QtWidgets.QPushButton("Take Photo")

        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(10, 100)
        self.speed_slider.setValue(self.speed)
        self.speed_label = QtWidgets.QLabel(f"Speed: {self.speed}")

        self.state_label = QtWidgets.QLabel("State: disconnected")
        self.battery_label = QtWidgets.QLabel("Battery: -")
        self.height_label = QtWidgets.QLabel("Height: -")
        self.wifi_label = QtWidgets.QLabel("WiFi: -")

        help_text = QtWidgets.QLabel(
            "Keys:\n"
            "W/S: Forward/Back\n"
            "Q/E: Left/Right\n"
            "R/F: Up/Down\n"
            "A/D: Yaw\n"
            "T: Takeoff, L: Land\n"
            "Space: Hover"
        )
        help_text.setStyleSheet("color:#bcccdc;")

        for w in [
            title,
            self.btn_connect,
            self.btn_disconnect,
            self.btn_takeoff,
            self.btn_land,
            self.btn_emergency,
            self.btn_photo,
            self.speed_label,
            self.speed_slider,
            self.state_label,
            self.battery_label,
            self.height_label,
            self.wifi_label,
            help_text,
        ]:
            side_layout.addWidget(w)
        side_layout.addStretch(1)

        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(side)

    def _bind_signals(self) -> None:
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_takeoff.clicked.connect(self._takeoff)
        self.btn_land.clicked.connect(self._land)
        self.btn_emergency.clicked.connect(self._emergency)
        self.btn_photo.clicked.connect(self._take_photo)
        self.speed_slider.valueChanged.connect(self._set_speed)

    def _set_speed(self, v: int) -> None:
        self.speed = int(v)
        self.speed_label.setText(f"Speed: {self.speed}")

    def _connect(self) -> None:
        try:
            self.controller.connect(timeout=30.0)
            self.video.start()
            self.state_label.setText("State: connected")
        except Exception as exc:
            self.state_label.setText(f"State: connect failed ({exc})")

    def _disconnect(self) -> None:
        self.pressed_keys.clear()
        self.video.stop()
        self.controller.disconnect()
        self.state_label.setText("State: disconnected")

    def _takeoff(self) -> None:
        try:
            self.controller.takeoff()
        except Exception as exc:
            self.state_label.setText(f"State: takeoff failed ({exc})")

    def _land(self) -> None:
        try:
            self.controller.land()
        except Exception as exc:
            self.state_label.setText(f"State: land failed ({exc})")

    def _emergency(self) -> None:
        self.controller.emergency()

    def _take_photo(self) -> None:
        if self.latest_frame is None:
            self.state_label.setText("State: no frame to save")
            return
        os.makedirs("img/uav", exist_ok=True)
        path = os.path.join("img", "uav", f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(path, cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR))
        self.state_label.setText(f"State: photo saved {path}")

    def _control_tick(self) -> None:
        axis = {"fb": 0, "lr": 0, "ud": 0, "yaw": 0}
        for key in self.pressed_keys:
            if key in KEY_MAP:
                channel, sign = KEY_MAP[key]
                axis[channel] += sign * self.speed

        for channel in axis:
            axis[channel] = max(-100, min(100, axis[channel]))

        self.controller.update_axis(fb=axis["fb"], lr=axis["lr"], ud=axis["ud"], yaw=axis["yaw"])

    def _refresh_telemetry(self) -> None:
        s = self.controller.get_state_snapshot()
        self.battery_label.setText(f"Battery: {s.battery if s.battery >= 0 else '-'}")
        self.height_label.setText(f"Height: {s.height_cm if s.height_cm >= 0 else '-'} cm")
        self.wifi_label.setText(f"WiFi: {s.wifi if s.wifi >= 0 else '-'}")

    def _render_video(self) -> None:
        frame = self.video.get_latest_frame()
        if frame is None:
            if self.video.last_error:
                self.video_label.setText(f"Video error: {self.video.last_error}")
            return

        self.latest_frame = frame
        h, w, c = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, c * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        k = event.key()
        if k in KEY_MAP:
            self.pressed_keys.add(k)
        elif k == QtCore.Qt.Key_T:
            self._takeoff()
        elif k == QtCore.Qt.Key_L:
            self._land()
        elif k == QtCore.Qt.Key_Space:
            self.pressed_keys.clear()
            self.controller.hover()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        k = event.key()
        if k in self.pressed_keys:
            self.pressed_keys.remove(k)
        super().keyReleaseEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.pressed_keys.clear()
        self.video.stop()
        self.controller.disconnect()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = TelloControlPanel()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
