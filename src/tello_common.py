from djitellopy import Tello
from djitellopy.tello import TelloException


def connect_tello_with_fallback(tello: Tello) -> str:
    try:
        tello.connect()
        return "state"
    except TelloException:
        try:
            tello.connect(wait_for_state=False)
        except TypeError:
            tello.connect(False)
        return "command_only"
