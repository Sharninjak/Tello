import datetime as dt
from djitellopy import Tello
from djitellopy.tello import TelloException


class SafeTello(Tello):
    def __del__(self):
        try:
            super().__del__()
        except Exception:
            pass


def safe_read(label: str, reader) -> str:
    try:
        return str(reader())
    except Exception as exc:
        return f"N/A ({label} failed: {exc})"


def main() -> int:
    session_prefix = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tello = None

    try:
        tello = SafeTello()
        tello.connect()
        print(f"battery: {safe_read('battery?', tello.get_battery)}")
        print(f"sdk: {safe_read('sdk?', tello.query_sdk_version)}")
        print(f"serial: {safe_read('sn?', tello.query_serial_number)}")
        print(f"state: {safe_read('state', tello.get_current_state)}")

        try:
            tello.streamon()
            print(f"streamon ok, session: {session_prefix}")
            return 0
        except TelloException as exc:
            print(f"streamon failed: {exc}")
            return 1
        except OSError as exc:
            print(f"streamon failed: {exc}")
            return 1
    finally:
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
    raise SystemExit(main())

