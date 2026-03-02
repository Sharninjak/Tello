import datetime as dt
import time
from pathlib import Path

import cv2
from djitellopy import Tello
from djitellopy.tello import TelloException

from tello_common import connect_tello_with_fallback


class SafeTello(Tello):
    def __del__(self):
        try:
            super().__del__()
        except Exception:
            pass


def start_stream_with_retry(tello: Tello, retries: int = 3):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            try:
                tello.streamoff()
            except Exception:
                pass

            tello.streamon()
            time.sleep(1.0)
            frame_reader = tello.get_frame_read()

            deadline = time.time() + 5.0
            while time.time() < deadline:
                frame = frame_reader.frame
                if frame is not None and getattr(frame, "size", 0) > 0:
                    return frame_reader
                time.sleep(0.02)

            raise RuntimeError("Video stream started but no frame received within 5s")
        except Exception as exc:
            last_error = exc
            print(f"[WARN] stream init failed (attempt {attempt}/{retries}): {exc}")
            time.sleep(0.8)

    raise TelloException(f"Failed to start video stream after {retries} attempts: {last_error}")


def main() -> None:
    output_dir = Path("img")
    output_dir.mkdir(parents=True, exist_ok=True)

    tello = None
    frame_reader = None

    writer = None
    recording = False
    video_path = None

    try:
        tello = SafeTello()
        mode = connect_tello_with_fallback(tello)
        print(f"[INFO] connected mode={mode}")

        frame_reader = start_stream_with_retry(tello, retries=3)
        print("[INFO] video stream is ready")

        while True:
            frame = frame_reader.frame
            if frame is None:
                time.sleep(0.01)
                continue

            now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, now_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            cv2.putText(
                frame,
                "p:photo  v:start/stop video  q:quit",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 220),
                2,
            )

            if recording and writer is not None:
                writer.write(frame)
                cv2.putText(frame, "REC", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Tello Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("p"):
                image_path = output_dir / f"image_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(image_path), frame)
                print(f"[INFO] saved image: {image_path}")
            if key == ord("v"):
                if not recording:
                    h, w = frame.shape[:2]
                    video_path = output_dir / f"video_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
                    recording = True
                    print(f"[INFO] recording started: {video_path}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"[INFO] recording stopped: {video_path}")

    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
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


if __name__ == "__main__":
    main()
