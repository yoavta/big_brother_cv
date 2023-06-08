import pytest
import cv2
from main import load_yolo, get_objects, detector, append_objs_to_img
import os

# Load YOLO model
model, classes, colors, output_layers = load_yolo()
model_loaded = [model, classes, colors, output_layers]

def process_image(image_path):
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (640, 480))
    cv2_im = frame.copy()

    threshold = 0.6
    objs = get_objects(frame, threshold, model_loaded, cv2_im)

    # show image
    hands, cv2_im = detector.findHands(cv2_im)
    if hands is not None:
        for hand in hands:
            objs.append(('hand', hand))

    moving_sensitivity = 140
    rectangle_size = moving_sensitivity
    screen_width = 640
    screen_height = 480

    half_rec_size = int(rectangle_size / 2)
    center_screen = (int(screen_width / 2), int(screen_height / 2))
    cv2.circle(cv2_im, center_screen, 2, (0, 0, 255), cv2.FILLED)
    top_left = (center_screen[0] - half_rec_size, center_screen[1] - half_rec_size)

    append_objs_to_img(cv2_im, objs)

    cv2.imshow("Image", cv2_im)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

    return objs


@pytest.mark.parametrize("image_path", [f'tests/test_images/{img}' for img in os.listdir("tests/test_images")])
def test_detect_hands(image_path):
    objs = process_image(image_path)
    hand_objs = [obj for obj in objs if obj[0] == 'hand']
    assert len(hand_objs) > 0, f"No hands detected in {image_path}"


@pytest.mark.parametrize("image_path", [f'tests/test_images/{img}' for img in os.listdir("tests/test_images")])
def test_detect_people(image_path):
    objs = process_image(image_path)
    people_objs = [obj for obj in objs if obj[0] == 'person']
    assert len(people_objs) > 0, f"No people detected in {image_path}"
