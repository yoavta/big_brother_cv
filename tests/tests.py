import os

import cv2
import pytest

from src.DetectorStore import DetectorStore
from src.main import camera_move_check

models_store = DetectorStore()
root_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(root_dir, "test_images")

is_debug = False


@pytest.fixture(autouse=True)
def setup_and_teardown():
    yield  # this is where the testing happens

    # Code that will be executed after each test
    print("Teardown")


@pytest.mark.parametrize("image_path", [f'test_images/{img}' for img in os.listdir(image_path)])
def test_detect_hands(image_path):
    split_path = image_path.split('/')[1].split('.')[0].split('-')
    image_path = os.path.join(root_dir, image_path)


    if is_debug:
        cv2.imshow('img', cv2.imread(image_path))
        cv2.waitKey(0)

    img = cv2.imread(image_path)

    hands, img_res = models_store.hand_detector.get_model().findHands(img)

    if is_debug:
        cv2.imshow('img', img_res)
        cv2.waitKey(0)

    num_of_expected_hands = split_path.count('h')
    assert len(
        hands) == num_of_expected_hands, f"Expected {num_of_expected_hands} hands, but found {len(hands)} in {image_path}"


@pytest.mark.parametrize("image_path", [f'test_images/{img}' for img in os.listdir(image_path)])
def test_detect_people(image_path):

    split_path = image_path.split('/')[1].split('.')[0].split('-')

    if is_debug:
        cv2.imshow('img', cv2.imread(image_path))
        cv2.waitKey(0)

    img = cv2.imread(image_path)
    blob, outputs = models_store.object_detector.detect_objects(img)

    if is_debug:
        cv2.imshow('img', img)
        cv2.waitKey(0)

    is_person_expected = split_path.count('p') > 0
    is_person_found = len(outputs) > 0
    assert is_person_expected == is_person_found, f"Expected {is_person_expected} people, but found {is_person_found} in {image_path}"


# test if the move is correct
@pytest.mark.parametrize("image_path", [f'test_images/{img}' for img in os.listdir(image_path)])
def test_detect_move(image_path):
    split_path = image_path.split('/')[1].split('.')[0].split('-')
    image_path = os.path.join(root_dir, image_path)
    img = cv2.imread(image_path)
    rectangle_size = 100



    camera_move_check(img,rectangle_size,)