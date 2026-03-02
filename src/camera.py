import atexit
import time

import cv2
from djitellopy import Tello

from tello_common import connect_tello_with_fallback


class SafeTello(Tello):
    def __del__(self):
        try:
            super().__del__()
        except Exception:
            pass


def start_stream_with_retry(tello: Tello, retries: int = 2, first_frame_timeout: float = 4.0):
    last_error = None

    for attempt in range(1, retries + 1):
        frame_reader = None
        try:
            try:
                tello.streamoff()
            except Exception:
                pass

            tello.streamon()
            time.sleep(0.8)
            frame_reader = tello.get_frame_read()

            deadline = time.time() + first_frame_timeout
            while time.time() < deadline:
                frame = frame_reader.frame
                if frame is not None and getattr(frame, "size", 0) > 0:
                    return frame_reader
                time.sleep(0.02)

            raise RuntimeError(f"Video stream started but no frame received within {first_frame_timeout}s")
        except Exception as exc:
            last_error = exc
            print(f"[WARN] stream init failed (attempt {attempt}/{retries}): {exc}")
            try:
                if frame_reader is not None and hasattr(frame_reader, "stop"):
                    frame_reader.stop()
            except Exception:
                pass
            time.sleep(0.8)

    raise RuntimeError(f"Failed to start video stream after {retries} attempts: {last_error}")


def safe_shutdown(tello: Tello | None, frame_reader) -> None:
    if frame_reader is not None:
        try:
            frame_reader.stop()
        except Exception:
            pass

    if tello is not None:
        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def main() -> None:
    tello = None
    frame_reader = None
    last_valid_frame_bgr = None
    last_frame_ok_ts = 0.0
    frame_stale_timeout = 1.6
    last_reconnect_try_ts = 0.0
    reconnect_interval = 0.8
    cleanup_done = False

    def cleanup_once():
        nonlocal cleanup_done, tello, frame_reader
        if cleanup_done:
            return
        cleanup_done = True
        safe_shutdown(tello, frame_reader)

    atexit.register(cleanup_once)

    try:
        tello = SafeTello()
        mode = connect_tello_with_fallback(tello)
        print(f"[INFO] connected mode={mode}")

        print("[INFO] camera test started. Press q to quit.")
        window_name = "Tello Camera Test"

        while True:
            now = time.time()

            if frame_reader is None and (now - last_reconnect_try_ts) >= reconnect_interval:
                last_reconnect_try_ts = now
                try:
                    frame_reader = start_stream_with_retry(tello, retries=2, first_frame_timeout=3.5)
                    last_frame_ok_ts = time.time()
                    print("[INFO] video stream connected")
                except Exception as exc:
                    print(f"[WARN] reconnect failed: {exc}")

            frame = None if frame_reader is None else frame_reader.frame
            if frame is not None and getattr(frame, "size", 0) > 0:
                try:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    frame_bgr = frame

                last_valid_frame_bgr = frame_bgr.copy()
                last_frame_ok_ts = now
                cv2.imshow(window_name, frame_bgr)
            elif last_valid_frame_bgr is not None:
                cv2.imshow(window_name, last_valid_frame_bgr)

            if frame_reader is not None and (now - last_frame_ok_ts) > frame_stale_timeout:
                print("[WARN] video stream stale, reconnecting...")
                try:
                    if hasattr(frame_reader, "stop"):
                        frame_reader.stop()
                except Exception:
                    pass
                frame_reader = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user")
    except Exception as exc:
        print(f"[ERROR] camera test runtime error: {exc}")
    finally:
        cleanup_once()


if __name__ == "__main__":
    main()
