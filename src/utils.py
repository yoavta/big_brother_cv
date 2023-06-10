import os
import re
from subprocess import call

import cv2

from src import Bbox
from src.controller_client import ActionsTypes


def speak(text):
    cmd_beg = 'say "'
    cmd_end = '"'
    call([cmd_beg + text + cmd_end], shell=True)


def read_label_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
            ret[int(pair[0])] = pair[1].strip()
        else:
            ret[row_number] = content.strip()
    return ret


def read_file(file_path):
    with open(file_path, 'rt') as f:
        return f.read().rstrip('\n').split('\n')


def get_path(relative_path):
    return os.path.join(os.path.dirname(__file__), relative_path)


def camera_move_check(img, rectangle_size, center_object, person_box: Bbox, screen_width, screen_height) -> ActionsTypes:
    half_rec_size = int(rectangle_size / 2)
    center_screen = (int(screen_width / 2), int(screen_height / 2))
    top_left = (center_screen[0] - half_rec_size, center_screen[1] - half_rec_size)

    cv2.rectangle(img, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)
    x, y = center_object[0], center_object[1]
    left_x = top_left[0]
    right_x = top_left[0] + rectangle_size

    frame_size = screen_width * screen_height
    person_size = (person_box.xmax - person_box.xmin) * (person_box.ymax - person_box.ymin)
    if person_size > frame_size * 0.3:
        return ActionsTypes.BACKWARD

    if person_size < frame_size * 0.1:
        return ActionsTypes.FORWARD

    if x < left_x:
        return ActionsTypes.LEFT

    if x > right_x:
        return ActionsTypes.RIGHT
