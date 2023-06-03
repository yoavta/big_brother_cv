import socket
import time
import cv2

import numpy as np

host = '10.0.2.235'
port = 12345


def receive_data(conn):
    data_length = 921600
    need_to_receive = data_length
    chunk_size = 3600

    received_data = b''
    while len(received_data) < data_length and need_to_receive > chunk_size:
        data = conn.recv(chunk_size)
        if not data:
            break
        received_data += data
        need_to_receive -= len(data)

    while need_to_receive > 0:
        data = conn.recv(need_to_receive)
        received_data += data
        need_to_receive -= len(data)

    return received_data


def init_server():
    global server_socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f'Server listening on {host}:{port}')


def accept_connection():
    global conn, addr
    conn, addr = server_socket.accept()
    print(f'Connection from {addr}')


def process_data():
    print(f'Processing data... time {time.time()}')
    global conn
    received_data = receive_data(conn)
    if not received_data:
        return

    print(f'Received data... time {time.time()}')
    np_array = np.frombuffer(received_data, dtype=np.uint8)
    image_shape = (480, 640)
    total_size = image_shape[0] * image_shape[1] * 3
    if len(np_array) < total_size:
        print(f'Not enough data... time {time.time()}')
        print(f'len(np_array) = {len(np_array)}')
        print(f'total_size = {total_size}')
        return
    image_data = np_array[:total_size]
    image = image_data.reshape(image_shape[0], image_shape[1], 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([50, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Convert back to RGB
    image = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    return image


def run_server():
    init_server()

#
# def test():
#     np_arr = np.load('temp.npy')
#     image_shape = (480, 640)
#     total_size = image_shape[0] * image_shape[1] * 3
#     image_data = np_arr[:total_size]
#     image = image_data.reshape(image_shape[0], image_shape[1], -1)
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
#
# test()
