import socket
from enum import Enum

host = '10.0.2.128'
port = 12347


class ActionsTypes(Enum):
    LEFT = "left",
    RIGHT = "right",
    STOP = "stop",
    START = "start",
    TURN_ON = "turn_on",
    TURN_OFF = "turn_off"


def send_data(action: ActionsTypes):
    print(f"Action: {action}")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    data = str(action.value[0]).encode()
    client_socket.sendall(data)

    client_socket.close()
