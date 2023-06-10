import os

import cv2
import pytest

from src.DetectorStore import DetectorStore
from src.categories_list import CategoriesList
from src.data_proceesor import analyze_connections
from src.firebase_config import FirebaseConfig
from src.utils import camera_move_check

detectors_store = DetectorStore()
firebase = FirebaseConfig()
root_dir = os.path.dirname(os.path.abspath(__file__))
image_path1 = os.path.join(root_dir, "test_images_1")
image_path2 = os.path.join(root_dir, "test_images_2")
image_path3 = os.path.join(root_dir, "test_images_3")
image_path4 = os.path.join(root_dir, "test_images_4")
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 0, 0)  # white color
thickness = 2

is_debug = True


@pytest.fixture(autouse=True)
def setup_and_teardown():
    yield  # this is where the testing happens

    # Code that will be executed after each test
    print("Teardown")


@pytest.mark.parametrize("image_path", [f'test_images_1/{img}' for img in os.listdir(image_path1)])
def test_detect_hands(image_path):
    split_path = image_path.split('/')[1].split('.')[0].split('-')
    image_path = os.path.join(root_dir, image_path)
    img = cv2.imread(image_path)
    assert img is not None, f"Could not read image {image_path}"
    hands, img_res = detectors_store.hand_detector.get_model().findHands(img)

    num_of_expected_hands = split_path.count('h')
    if is_debug:
        cv2.putText(img_res, f"Expected: {num_of_expected_hands}, Found: {len(hands)}", (10, 30), font, font_scale,
                    color, thickness)
        cv2.imshow('img', img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    assert len(
        hands) == num_of_expected_hands, f"Expected {num_of_expected_hands} hands, but found {len(hands)} in {image_path}"


@pytest.mark.parametrize("image_path", [f'test_images_1/{img}' for img in os.listdir(image_path1)])
def test_detect_people(image_path):
    split_path = image_path.split('/')[1].split('.')[0].split('-')
    image_path = os.path.join(root_dir, image_path)
    img = cv2.imread(image_path)
    assert img is not None, f"Could not read image {image_path}"

    frame = img.copy()
    obj = detectors_store.object_detector.get_objects(frame, 0.5, img)

    is_person_expected = split_path.count('p') > 0
    is_person_found = len(obj) > 0

    if is_debug:
        img_with_title = cv2.putText(img, f"Expected: {int(is_person_expected)}, Found: {int(is_person_found)}",
                                     (50, 50),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('img', img_with_title)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    assert is_person_expected == is_person_found, f"Expected {is_person_expected} people, but found {is_person_found} in {image_path}"


# test if the move is correct
@pytest.mark.parametrize("image_path", [f'test_images_2/{img}' for img in os.listdir(image_path2)])
def test_detect_move(image_path):
    split_path = image_path.split('/')[1].split('.')[0].split('-')
    image_path = os.path.join(root_dir, image_path)
    frame = cv2.imread(image_path)
    cv2_im = frame.copy()
    image_height, image_width, _ = frame.shape
    threshold = 0.5
    objs = detectors_store.object_detector.get_objects(frame, threshold, cv2_im)
    hands = []
    bboxes = []
    class_ids = []
    possible_connections = {}
    cv2_im, person_center, person_box = detectors_store.append_objs_to_img(cv2_im, objs, class_ids, bboxes, hands,
                                                                           possible_connections)

    rectangle_size = 100
    res = camera_move_check(cv2_im, rectangle_size, person_center, person_box, image_width, image_height)
    if is_debug:
        img_with_title = cv2.putText(cv2_im, str(res), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('img', img_with_title)
        cv2.waitKey(0)
    first_letter = 'N' if not res else str(res).split('.')[1][0]
    assert first_letter == split_path[0], f"Expected {split_path[0]} but got {first_letter}"


# test overlap detection
@pytest.mark.parametrize("image_path", [f'test_images_3/{img}' for img in os.listdir(image_path3)])
def test_detect_overlap(image_path):
    split_path = image_path.split('/')[1].split('.')[0].split('-')
    image_path = os.path.join(root_dir, image_path)
    frame = cv2.imread(image_path)
    cv2_im = frame.copy()
    image_height, image_width, _ = frame.shape
    threshold = 0.2
    objs = detectors_store.object_detector.get_objects(frame, threshold, cv2_im)
    hands = []
    bboxes = []
    class_ids = []
    possible_connections = {}
    detectors_store.append_objs_to_img(cv2_im, objs, class_ids, bboxes, hands,
                                       possible_connections)

    categories = CategoriesList(firebase)
    events = analyze_connections(possible_connections, None, 'test_user', categories, None, is_speak=False)

    if split_path[0] == "V":
        assert len(events) > 0, f"Expected {split_path[0]} events but goﬁt {len(events)}"
    elif split_path[0] == "c":
        assert len(events) > 0, f"Expected {split_path[0]} events but got {len(events)}"
    else:
        assert len(events) == 0, f"Expected {split_path[0]} events but got {len(events)}"

# test overlap hands
@pytest.mark.parametrize("image_path", [f'test_images_4/{img}' for img in os.listdir(image_path4)])
def test_detect_overlap_hands(image_path):
    split_path = image_path.split('/')[1].split('.')[0].split('-')
    image_path = os.path.join(root_dir, image_path)
    frame = cv2.imread(image_path)
    cv2_im = frame.copy()
    image_height, image_width, _ = frame.shape
    threshold = 0.2
    objs = detectors_store.object_detector.get_objects(frame, threshold, cv2_im)
    hands, img_res = detectors_store.hand_detector.get_model().findHands(cv2_im)
    bboxes = []
    class_ids = []
    possible_connections = {}
    detectors_store.append_objs_to_img(img_res, objs, class_ids, bboxes, hands,
                                       possible_connections)
    categories = CategoriesList(firebase)
    events = analyze_connections(possible_connections, None, 'test_user', categories, None, is_speak=False)
    assert len(events) == 1, f"Expected {split_path[0]} events but got {len(events)}"
    if is_debug:
        cv2.imshow('img', img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    pass


