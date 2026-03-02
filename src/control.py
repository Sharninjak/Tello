import os
import ctypes
import threading
import time
from datetime import datetime
from dataclasses import dataclass

import cv2
import numpy as np
from djitellopy import Tello

from tello_common import connect_tello_with_fallback

user32 = ctypes.windll.user32
KEY_DOWN_MASK = 0x8000

KEY_CODE_MAP = {
    "w": 0x57,
    "s": 0x53,
    "a": 0x41,
    "d": 0x44,
    "q": 0x51,
    "e": 0x45,
    "r": 0x52,
    "shift": 0x10,
    "g": 0x47,
    "l": 0x4C,
    "backtick": 0xC0,
    "esc": 0x1B,
}

KEY_BINDINGS = {
    "axes": {
        "forward": "w",
        "backward": "s",
        "left": "q",
        "right": "e",
        "yaw_left": "a",
        "yaw_right": "d",
        "up": "r",
        "down": "shift",
    },
    "actions": {
        "takeoff": "backtick",
        "land": "l",
        "photo": "g",
        "quit": "esc",
    },
}

PRESSED_KEYS_ORDER = [
    KEY_BINDINGS["axes"]["forward"],
    KEY_BINDINGS["axes"]["backward"],
    KEY_BINDINGS["axes"]["left"],
    KEY_BINDINGS["axes"]["right"],
    KEY_BINDINGS["axes"]["yaw_left"],
    KEY_BINDINGS["axes"]["yaw_right"],
    KEY_BINDINGS["axes"]["up"],
    KEY_BINDINGS["axes"]["down"],
    KEY_BINDINGS["actions"]["takeoff"],
    KEY_BINDINGS["actions"]["land"],
    KEY_BINDINGS["actions"]["photo"],
]


def key_to_label(key_name: str) -> str:
    label_map = {
        "backtick": "`",
        "shift": "SHIFT",
        "esc": "ESC",
    }
    return label_map.get(key_name, key_name.upper())


def get_key_guide_lines() -> tuple:
    axes = KEY_BINDINGS["axes"]
    actions = KEY_BINDINGS["actions"]
    return (
        f"{key_to_label(axes['forward'])}/{key_to_label(axes['backward'])} : Forward / Backward",
        f"{key_to_label(axes['left'])}/{key_to_label(axes['right'])} : Move Left / Move Right",
        f"{key_to_label(axes['yaw_left'])}/{key_to_label(axes['yaw_right'])} : Yaw Left / Yaw Right",
        f"{key_to_label(axes['up'])}/{key_to_label(axes['down'])}: Move Up / Move Down",
        f"{key_to_label(actions['takeoff'])} / {key_to_label(actions['land'])}  : Takeoff / Land",
        f"{key_to_label(actions['photo'])}      : Take Photo",
        f"{key_to_label(actions['quit'])}    : Quit",
    )


_bound_keys = set(KEY_BINDINGS["axes"].values()) | set(KEY_BINDINGS["actions"].values())
VK_MAP = {key: KEY_CODE_MAP[key] for key in _bound_keys}

def is_pressed(key: str) -> bool:
    """
    Description
    
    Args:
        param_name [type]: Description
    
    Returns:
        type: Description
    """
    
    vk = VK_MAP.get(key.lower())
    if vk is None:
        return False
    return (user32.GetAsyncKeyState(vk) & KEY_DOWN_MASK) != 0

@dataclass(frozen=True)
class PanelUIConfig:

    # 窗口名称配置
    window_name: str = "Tello Control Panel"  # 控制面板窗口的名称

    
    # 面板信息宽度配置
    panel_info_width: int = 340  # 控制面板的信息宽度
    padding_x: int = 16  # 水平方向的内边距
    panel_bg: int = 24  # 面板背景颜色值
    
    # 相机高度配置
    min_camera_height: int = 480  # 相机显示的最小高度
    max_camera_height: int = 720  # 相机显示的最大高度
    
    # 文本缩放比例配置
    title_scale: float = 0.70  # 标题文本的缩放比例
    header_scale: float = 0.60  # 头部文本的缩放比例
    line_scale: float = 0.52  # 普通行文本的缩放比例
    value_scale: float = 0.52  # 数值文本的缩放比例
    
    # 布局和刷新配置
    line_gap: int = 26  # 文本行之间的间隔
    refresh_hz: float = 30.0  # 界面刷新频率（赫兹）
    
    # 键盘操作指南配置
    key_guide_lines: tuple = get_key_guide_lines()

class PerceptionPanelWorker:
    """
    Class Description
    
    Attributes:
        attr_name [type]: Description
    
    Methods:
        __init__: INITIAL
    """
    def __init__(self, frame_read, ui_config: PanelUIConfig):
        self.frame_read = frame_read                # 指向Tello的frame_read对象
        self.ui = ui_config                         # 面板UI配置对象，包含各种显示参数
        self.window_name = self.ui.window_name      # 窗口名称，从UI配置中获取
        self._state_lock = threading.Lock()         # 线程锁，用于保护状态数据的访问
        self._state = {                             # 控制指令
            "fb": 0, "lr": 0, "ud": 0, "yaw": 0,     
            "battery": 0, "height": 0
        }
        self._running = False                       # 运行状态标志
        self._thread = None                         # 工作线程对象

    def start(self):
        # 检查是否已经在运行中
        if self._running: return
        # 设置运行状态为True
        self._running = True
        # 创建并启动一个守护线程，用于执行_loop方法
        # 线程名称为"PerceptionPanelWorker"，设置为守护线程意味着当主程序结束时，该线程也会随之结束
        self._thread = threading.Thread(target=self._loop, name="PerceptionPanelWorker", daemon=True)
        # 启动线程
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        cv2.destroyAllWindows()

    def update_state(self, fb: int, lr: int, ud: int, yaw: int, battery: int, height: int):
        """
        Description
        
        Args:
            param_name [type]: Description
        
        Returns:
            type: Description
        """
        
        with self._state_lock:
            self._state["fb"] = fb
            self._state["lr"] = lr
            self._state["ud"] = ud
            self._state["yaw"] = yaw
            self._state["battery"] = battery
            self._state["height"] = height

    def _loop(self):
        """
        Description
        
        Args:
            param_name [type]: Description
        
        Returns:
            type: Description
        """
        
        # 创建一个自动调整大小的窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        # 当程序运行标志为真时，持续执行循环
        while self._running:
            # djitellopy usually processes background frame read in RGB format
            frame_rgb = self.frame_read.frame
            # 检查frame_rgb是否为None或高度为0（即没有有效图像帧）
            if frame_rgb is None or frame_rgb.shape[0] == 0:
                # 创建一个指定大小的黑色图像，尺寸为最小相机高度和对应的宽度（4:3比例）
                # 图像数据类型为8位无符号整数，像素值设为40（深灰色）
                image_bgr = np.full((self.ui.min_camera_height, int(self.ui.min_camera_height * 4 / 3), 3), 40, dtype=np.uint8)
                cv2.putText(image_bgr, "Waiting for camera...", (60, self.ui.min_camera_height // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            else:
                image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            image_bgr = self._resize_camera_keep_aspect(image_bgr)
            info_panel = self._build_info_panel(image_bgr.shape[0])
            canvas = np.hstack([info_panel, image_bgr])
            
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
            time.sleep(1.0 / self.ui.refresh_hz)

    def _resize_camera_keep_aspect(self, image_bgr):
        """
        Description
        
        Args:
            param_name [type]: Description
        
        Returns:
            type: Description
        """
        
        h, w = image_bgr.shape[:2] # 获取图像的高度和宽度
        if h <= 0 or w <= 0: return image_bgr
        target_h = int(max(self.ui.min_camera_height, min(self.ui.max_camera_height, h))) # 计算目标高度，确保在最小和最大相机高度之间
        if target_h == h: return image_bgr
        scale = target_h / float(h)
        target_w = max(1, int(round(w * scale)))
        return cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _build_info_panel(self, panel_height):
        """
        Description
        
        Args:
            param_name [type]: Description
        
        Returns:
            type: Description
        """
        
        panel = np.full((panel_height, self.ui.panel_info_width, 3), self.ui.panel_bg, dtype=np.uint8)
        
        cv2.putText(panel, "Tello Control Panel", (self.ui.padding_x, 34), cv2.FONT_HERSHEY_SIMPLEX, self.ui.title_scale, (220, 220, 220), 2)
        cv2.putText(panel, "Key Mappings:", (self.ui.padding_x, 70), cv2.FONT_HERSHEY_SIMPLEX, self.ui.header_scale, (170, 220, 170), 2)

        y = 100
        for line in self.ui.key_guide_lines:
            cv2.putText(panel, line, (self.ui.padding_x, y), cv2.FONT_HERSHEY_SIMPLEX, self.ui.line_scale, (235, 235, 235), 1)
            y += self.ui.line_gap

        with self._state_lock:
            fb = self._state["fb"]
            lr = self._state["lr"]
            ud = self._state["ud"]
            yaw = self._state["yaw"]
            battery = self._state["battery"]
            height = self._state["height"]

        cv2.putText(panel, "Pressed Keys:", (self.ui.padding_x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, self.ui.header_scale, (170, 220, 170), 2)
        y += 40
        pressed = []
        for key_name in PRESSED_KEYS_ORDER:
            if is_pressed(key_name):
                pressed.append(key_to_label(key_name))
        cv2.putText(panel, ", ".join(pressed) if pressed else "(none)", (self.ui.padding_x, y), cv2.FONT_HERSHEY_SIMPLEX, self.ui.value_scale, (245, 225, 120), 2)
        y += 40

        cv2.putText(panel, "Current Command:", (self.ui.padding_x, y), cv2.FONT_HERSHEY_SIMPLEX, self.ui.header_scale, (170, 220, 170), 2)
        y += 30
        cv2.putText(panel, f"FB={fb:>3}  LR={lr:>3}", (self.ui.padding_x, y), cv2.FONT_HERSHEY_SIMPLEX, self.ui.value_scale, (220, 220, 220), 2)
        y += 30
        cv2.putText(panel, f"UD={ud:>3}  YAW={yaw:>3}", (self.ui.padding_x, y), cv2.FONT_HERSHEY_SIMPLEX, self.ui.value_scale, (220, 220, 220), 2)
        y += 50
        
        cv2.putText(panel, f"Alt: {height} cm | Bat: {battery}%", (self.ui.padding_x, y), cv2.FONT_HERSHEY_SIMPLEX, self.ui.header_scale, (120, 220, 245), 2)
        return panel

class KeyboardFlightController:
    """
    Class Description
    
    Attributes:
        attr_name [type]: Description
    
    Methods:
        __init__: INITIAL
    """
    
    def __init__(self):
        self.tello = Tello()  # 创建Tello对象
        self.speed = 10  # 设置默认飞行速度
        self.target_takeoff_height_cm = 30  # 起飞后目标高度（厘米）
        self.panel_worker = None  # UI面板线程
        self.frame_read = None  # 视频帧读取器
        self.last_frame_ok_ts = 0.0  # 最近一次收到有效视频帧的时间
        self.frame_stale_timeout = 2.0  # 判定视频流中断的阈值（秒）
        self.t_latch = False  # 起飞按键锁存状态
        self.l_latch = False  # 降落按键锁存状态
        self.g_latch = False  # 拍照按键锁存状态

    def run(self):

        """
        运行主控制循环
        连接Tello、启动视频流、处理用户输入和控制飞行器
        """
        try:
            mode = connect_tello_with_fallback(self.tello)  # 尝试连接Tello
            print(f"[INFO] Connected to Tello. Mode: {mode}")

            self._configure_video_stream()
            self.tello.streamon()  # 启动视频流
            self.frame_read = self._start_frame_reader()  # 获取视频帧读取器
            self.last_frame_ok_ts = time.time()

            # 初始化并启动UI面板工作线程
            ui_config = PanelUIConfig()
            self.panel_worker = PerceptionPanelWorker(self.frame_read, ui_config)
            self.panel_worker.start()

            # 设置控制指令发送间隔
            send_interval = 0.05
            last_send = time.time()

            while True:
                if is_pressed(KEY_BINDINGS["actions"]["quit"]):  # 检测退出键按下
                    print("Exiting...")
                    break

                self._recover_video_stream_if_needed()
                self.handle_latched_keys()  # 处理需要锁存的按键
                lr, fb, ud, yaw = self.read_manual_inputs()  # 读取手动输入

                # 获取电池电量和高度信息
                battery = self.tello.get_battery()
                height = self.tello.get_distance_tof() if hasattr(self.tello, 'get_distance_tof') else self.tello.get_height()
                self.panel_worker.update_state(fb, lr, ud, yaw, battery, height)  # 更新UI面板状态

                # 定期发送控制指令
                now = time.time()
                if now - last_send >= send_interval:
                    self.tello.send_rc_control(lr, fb, ud, yaw)
                    last_send = now

                time.sleep(0.01)  # 短暂休眠以减少CPU使用
        except KeyboardInterrupt:  # 处理Ctrl+C中断
            print("\nCtrl+C detected. Exiting.")
        except Exception as e:
            print(f"[ERROR] Controller startup/runtime error: {e}")
        finally:
            self._safe_shutdown()

    def _safe_shutdown(self):
        if self.panel_worker is not None:
            try:
                self.panel_worker.stop()
            except Exception:
                pass

        if self.frame_read is not None and hasattr(self.frame_read, "stop"):
            try:
                self.frame_read.stop()
            except Exception:
                pass

        try:
            self.tello.send_rc_control(0, 0, 0, 0)  # 停止所有运动
        except Exception:
            pass

        try:
            self.tello.streamoff()  # 关闭视频流
        except Exception:
            pass

        try:
            self.tello.end()  # 结束连接
        except Exception:
            pass

    def read_manual_inputs(self):
        """
        Description
        
        Args:
            param_name [type]: Description
        
        Returns:
            type: Description
        """
        
        lr = fb = ud = yaw = 0  # 初始化所有方向速度为0
        axes = KEY_BINDINGS["axes"]

        # 根据按键设置对应方向的速度
        if is_pressed(axes["forward"]): fb = self.speed  # 前进
        if is_pressed(axes["backward"]): fb = -self.speed  # 后退
        if is_pressed(axes["left"]): lr = -self.speed  # 左移
        if is_pressed(axes["right"]): lr = self.speed  # 右移
        if is_pressed(axes["yaw_left"]): yaw = -self.speed  # 左偏航
        if is_pressed(axes["yaw_right"]): yaw = self.speed  # 右偏航
        if is_pressed(axes["up"]): ud = self.speed  # 上升
        if is_pressed(axes["down"]): ud = -self.speed  # 下降
        return lr, fb, ud, yaw

    def handle_latched_keys(self):
        """
        Description
        
        Args:
            param_name [type]: Description
        
        Returns:
            type: Description
        """
        
        actions = KEY_BINDINGS["actions"]

        # Takeoff
        if is_pressed(actions["takeoff"]) and not self.t_latch:
            self.t_latch = True
            print("Takeoff Command Sent")
            try:
                self.tello.takeoff()
                self._adjust_takeoff_height(self.target_takeoff_height_cm)
            except Exception as e: print(f"Takeoff error: {e}")
        elif not is_pressed(actions["takeoff"]):
            self.t_latch = False

        # Land
        if is_pressed(actions["land"]) and not self.l_latch:
            self.l_latch = True
            print("Land Command Sent")
            try: self.tello.land()
            except Exception as e: print(f"Land error: {e}")
        elif not is_pressed(actions["land"]):
            self.l_latch = False

        # Photo (G)
        if is_pressed(actions["photo"]) and not self.g_latch:
            self.g_latch = True
            self.take_photo()
        elif not is_pressed(actions["photo"]):
            self.g_latch = False

    def _configure_video_stream(self):
        try:
            self.tello.set_video_resolution(Tello.RESOLUTION_480P)
        except Exception:
            pass

        try:
            self.tello.set_video_fps(Tello.FPS_30)
        except Exception:
            pass

        try:
            self.tello.set_video_bitrate(Tello.BITRATE_AUTO)
        except Exception:
            pass

    def _start_frame_reader(self):
        try:
            frame_read = self.tello.get_frame_read(with_queue=True, max_queue_len=64)
        except TypeError:
            frame_read = self.tello.get_frame_read()
        time.sleep(0.5)
        return frame_read

    def _get_latest_valid_frame(self, timeout_sec: float = 1.0):
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            if self.frame_read is None:
                time.sleep(0.02)
                continue

            frame = self.frame_read.frame
            if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2 and frame.shape[0] > 0:
                self.last_frame_ok_ts = time.time()
                return frame
            time.sleep(0.02)
        return None

    def _restart_video_stream(self):
        print("[WARN] Video stream interrupted, restarting...")
        try:
            if self.frame_read is not None and hasattr(self.frame_read, "stop"):
                self.frame_read.stop()
        except Exception:
            pass

        try:
            self.tello.streamoff()
        except Exception:
            pass

        time.sleep(0.3)
        self._configure_video_stream()
        self.tello.streamon()
        self.frame_read = self._start_frame_reader()
        self.panel_worker.frame_read = self.frame_read
        self.last_frame_ok_ts = time.time()

    def _recover_video_stream_if_needed(self):
        frame = self._get_latest_valid_frame(timeout_sec=0.05)
        if frame is not None:
            return

        is_stopped = bool(getattr(self.frame_read, "stopped", False)) if self.frame_read is not None else True
        stale = (time.time() - self.last_frame_ok_ts) > self.frame_stale_timeout
        if is_stopped or stale:
            self._restart_video_stream()

    def _adjust_takeoff_height(self, target_cm: int):

        """
        调整无人机起飞高度的
        :param target_cm: 目标高度，单位厘米
        """
        time.sleep(0.8)  # 等待0.8秒，让无人机稳定
        try:
            # 尝试获取无人机当前高度，优先使用TOF传感器，如果没有则使用普通高度获取方法
            current = self.tello.get_distance_tof() if hasattr(self.tello, 'get_distance_tof') else self.tello.get_height()
        except Exception:
            return  # 如果获取高度失败，直接返回

        # 检查当前高度是否有效
        if not isinstance(current, int) or current <= 0:
            return

        # 计算目标高度与当前高度的差值
        delta = target_cm - current
        # 如果高度差小于20厘米，认为高度已经足够接近，不进行调整
        if abs(delta) < 20:
            return

        try:

            # 根据高度差调整高度
            if delta > 0:  # 如果目标高度高于当前高度，则上升
                self.tello.move_up(min(delta, 500))
            else:
                self.tello.move_down(min(abs(delta), 500))
        except Exception as e:
            print(f"Height adjust error: {e}")

    def take_photo(self):
        frame = self._get_latest_valid_frame(timeout_sec=1.0)
        if frame is None:
            print("[Warning] No frame available. Photo not saved.")
            return

        save_dir = os.path.join("img", "uav")
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"photo_{ts}.jpg")

        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr_frame)
        print(f"[Photo] Saved to {filepath}")

def main():
    controller = KeyboardFlightController()
    controller.run()

if __name__ == "__main__":
    main()
