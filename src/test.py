from __future__ import annotations

import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from tellopy_core import TelloController
from tellopy_video import TelloVideoStream


class VideoTestWindow(QtWidgets.QWidget):
    def __init__(self, video: TelloVideoStream, duration_sec: float = 8.0) -> None:
        super().__init__()
        self.video = video
        self.has_frame = False

        self.setWindowTitle("Tello PyQt5 Video Test")
        self.resize(980, 760)

        self.video_label = QtWidgets.QLabel("Waiting for video frame...")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#101417;color:#d9e2ec;")

        self.info_label = QtWidgets.QLabel("Press ESC to close early")
        self.info_label.setStyleSheet("color:#334e68;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label, stretch=1)
        layout.addWidget(self.info_label)

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.timeout.connect(self._render)
        self.render_timer.start(33)

        self.close_timer = QtCore.QTimer(self)
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.close)
        self.close_timer.start(int(duration_sec * 1000))

    def _render(self) -> None:
        frame = self.video.get_latest_frame()
        if frame is None:
            if self.video.last_error:
                self.info_label.setText(f"Decoder info: {self.video.last_error}")
            return

        self.has_frame = True
        h, w, c = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, c * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.render_timer.stop()
        event.accept()


def run_pyqt_video_test(video: TelloVideoStream, duration_sec: float = 8.0) -> bool:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    win = VideoTestWindow(video=video, duration_sec=duration_sec)
    win.show()
    app.exec_()

    if owns_app:
        app.quit()
    return win.has_frame


def main() -> int:
    controller = TelloController()
    video = TelloVideoStream(controller)

    try:
        controller.connect(timeout=30.0)
        print("[OK] connect")

        video.start()
        deadline = time.time() + 8.0
        got_frame = False
        while time.time() < deadline:
            frame = video.get_latest_frame()
            if frame is not None and frame.size > 0:
                print(f"[OK] video frame: {frame.shape[1]}x{frame.shape[0]}")
                got_frame = True
                break
            time.sleep(0.05)

        if not got_frame:
            print(f"[FAIL] no frame decoded, error={video.last_error or '-'}")
            return 1

        state = controller.get_state_snapshot()
        print(f"[OK] battery={state.battery} height={state.height_cm} wifi={state.wifi}")

        print("[INFO] launching PyQt5 video display test...")
        if run_pyqt_video_test(video, duration_sec=8.0):
            print("[OK] PyQt5 video display test passed")
            return 0

        print(f"[FAIL] PyQt5 video display test failed, error={video.last_error or '-'}")
        return 1
    except Exception as exc:
        print(f"[FAIL] runtime error: {exc}")
        return 1
    finally:
        video.stop()
        controller.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

