import tellopy
import av
import cv2
import time
import numpy as np
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


class TelloCamera:
    """Tello相机类，负责获取和处理视频流"""

    def __init__(self):
        self.Tello_frame = None  # 存储Tello传回来的图像
        self.last_error = None  # 最近一次错误信息
        self.container = None  # 解码视频流的容器
        self.vid_stream = None  # 视频流
        self.height = 0  # 视频高度
        self.width = 0  # 视频宽度
        self.video_process_thread = None  # 视频处理线程
        self.is_running = False  # 标记是否正在运行

    def connect(self):
        """连接到Tello无人机"""
        try:
            self.drone = tellopy.Tello()
            self.drone.connect()
            self.drone.wait_for_connection(60.0)
            print("成功连接到Tello无人机")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def start_video(self):
        """启动视频流"""
        try:
            # 获取视频流
            self.container = av.open(self.drone.get_video_stream())
            self.vid_stream = self.container.streams.video[0]
            self.height = self.vid_stream.height
            self.width = self.vid_stream.width
            print(f"视频流已启动，分辨率: {self.width}x{self.height}")

            # 启动视频处理线程
            self.video_process_thread = threading.Thread(target=self.__Video_process)
            self.video_process_thread.daemon = True
            self.is_running = True
            self.video_process_thread.start()

            return True
        except Exception as e:
            print(f"启动视频流失败: {e}")
            self.last_error = f"decode error: {e}"
            return False

    def get_video_frame(self):
        """返回当前的Tello图像"""
        return self.Tello_frame

    def __Video_process(self):
        """单独线程，一直解码并处理来自无人机的视频流"""
        frame_skip = 30  # 跳过前30帧，减少初始黑屏等待
        while self.is_running:
            try:
                for frame in self.container.decode(video=0):
                    if not self.is_running:
                        break

                    if 0 < frame_skip:
                        frame_skip = frame_skip - 1
                        continue

                    start_time = time.time()
                    # 将帧转换为numpy数组
                    self.Tello_frame = np.array(frame.to_image())

                    # 计算时间基准，控制帧率
                    if frame.time_base < 1.0 / 60:
                        time_base = 1.0 / 60
                    else:
                        time_base = frame.time_base

                    frame_skip = int((time.time() - start_time) / time_base)

            except Exception as e:
                print(f"视频处理错误: {e}")
                self.last_error = str(e)
                break

    def stop(self):
        """停止视频流和连接"""
        self.is_running = False
        if self.video_process_thread and self.video_process_thread.is_alive():
            self.video_process_thread.join(timeout=2)

        if self.container:
            self.container.close()

        if hasattr(self, 'drone'):
            self.drone.quit()

        print("相机已停止")


class CameraWindow(QMainWindow):
    """相机显示窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Tello相机测试')
        self.setGeometry(100, 100, 960, 720)

        # 创建中央部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label)

        # 创建Tello相机实例
        self.camera = TelloCamera()

        # 创建定时器用于更新视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 约33fps

        # 连接相机
        self.connect_camera()

        self.channel = 3  # RGB通道数

    def connect_camera(self):
        """连接到相机"""
        print("正在连接到Tello无人机...")
        print("请确保已连接到Tello的WiFi网络")

        if self.camera.connect():
            if self.camera.start_video():
                print("相机已成功启动")
            else:
                print("启动视频流失败")
        else:
            print("连接失败")

    def update_frame(self):
        """更新视频帧显示"""
        frame = self.camera.get_video_frame()

        if self.camera.last_error:
            self.video_label.setText(f'视频错误: {self.camera.last_error}')
            return
        elif frame is None:
            self.video_label.setText('等待视频流...')
            return
        else:
            # 调整图像大小以适应显示窗口
            img_w = self.video_label.width()
            img_h = self.video_label.height()

            if img_w > 0 and img_h > 0:
                frame = cv2.resize(frame, (img_w, img_h))

                # 转换为QPixmap并显示
                pixmap = QPixmap.fromImage(
                    QImage(frame.data, img_w, img_h, 
                           self.channel * img_w, 
                           QImage.Format_RGB888)
                )
                self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.timer.stop()
        self.camera.stop()
        event.accept()


def main():
    """主函数"""
    import sys

    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
