import time

import cv2
import numpy as np

from categories_list import CategoriesList
from configuration import Configuration
from controller_client import ActionsTypes
from controller_client import send_data
from controller_server import run_server, accept_connection, process_data
from firebase_config import FirebaseConfig
from form import form
from src import Bbox
from src.Bbox import get_box_dimensions, BBox
from src.DetectorStore import DetectorStore
from src.Object import Object
from src.data_proceesor import analyze_connections
from src.image_utils import draw_labels
from utils import speak

configuration = Configuration()


# analyze connections with hands. check if there is overlapping between objects and hands.
def hand_connections(objs, hands, bboxes, class_ids, possible_connections):
    for hand in hands:
        current_frame_connections = []

        x2, y2, w2, h2 = hand['bbox'][0], hand['bbox'][1], hand['bbox'][2], hand['bbox'][3]
        total_cons = len(bboxes)
        last_cons = len(objs)
        if last_cons == 0 or total_cons == 0:
            break
        for j in range(total_cons):
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = class_ids[j]
            #             print(labels.get(obj_id,obj_id))
            #             if labels.get(obj_id,obj_id) == "bottle":
            #                 print("this")
            if configuration.labels.get(obj_id, obj_id) in configuration.hand_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    pass
                else:
                    if configuration.labels.get(obj_id, obj_id) in current_frame_connections:
                        continue
                    current_frame_connections.append(configuration.labels.get(obj_id, obj_id))
                    if configuration.labels.get(obj_id, obj_id) in possible_connections:
                        possible_connections[configuration.labels.get(obj_id, obj_id)] = \
                            possible_connections[configuration.labels.get(obj_id, obj_id)] + 1
                    else:
                        possible_connections[configuration.labels.get(obj_id, obj_id)] = 1


# analyze connections with person. check if there is overlapping between objects and person.
def person_connections(person_box: BBox, bboxes, class_ids, possible_connections):
    if person_box is not None:
        current_frame_connections = []
        x2, y2, w2, h2 = [person_box.xmin, person_box.ymin, person_box.xmax, person_box.ymax]
        total_cons = len(bboxes)
        for j in range(total_cons):
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = class_ids[j]
            if configuration.labels.get(obj_id, obj_id) in configuration.person_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    pass
                else:
                    if configuration.labels.get(obj_id, obj_id) in current_frame_connections:
                        continue
                    current_frame_connections.append(configuration.labels.get(obj_id, obj_id))
                    if configuration.labels.get(obj_id, obj_id) in possible_connections:
                        possible_connections[configuration.labels.get(obj_id, obj_id)] = \
                            possible_connections[configuration.labels.get(obj_id, obj_id)] + 1
                    else:
                        possible_connections[configuration.labels.get(obj_id, obj_id)] = 1


def get_objects(frame, threshold, object_detector, img):
    height, width, channels = frame.shape
    model, classes, colors, output_layers = object_detector.get_model()
    blob, outputs = object_detector.detect_objects(frame)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    draw_labels(boxes, colors, class_ids, classes, img, indexes)

    def make(bbox, score, id):
        return Object(id, score, BBox(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

    if indexes is None:
        return []

    objs = [make(bbox, score, class_ids[i]) for i, (bbox, score) in enumerate(zip(boxes, confs))]

    if isinstance(indexes, np.ndarray):
        flattened_indexes = indexes.flatten()
    else:
        flattened_indexes = np.array(indexes).flatten()

    filtered_objs = [objs[i] for i in flattened_indexes if objs[i].score > threshold]

    return filtered_objs


def camera_move_check(img, rectangle_size, center_object, person_box: Bbox) -> ActionsTypes:
    half_rec_size = int(rectangle_size / 2)
    center_screen = (int(configuration.screen_width / 2), int(configuration.screen_height / 2))
    top_left = (center_screen[0] - half_rec_size, center_screen[1] - half_rec_size)

    cv2.rectangle(img, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)
    x, y = center_object[0], center_object[1]
    left_x = top_left[0]
    right_x = top_left[0] + rectangle_size

    frame_size = configuration.screen_width * configuration.screen_height
    person_size = (person_box.xmax - person_box.xmin) * (person_box.ymax - person_box.ymin)
    if person_size > frame_size * 0.3:
        return ActionsTypes.BACKWARD

    if person_size < frame_size * 0.1:
        return ActionsTypes.FORWARD

    if x < left_x:
        return ActionsTypes.LEFT

    if x > right_x:
        return ActionsTypes.RIGHT


def append_objs_to_img(cv2_image, objs, class_ids, bboxes, hands, possible_connections):
    person_center = None
    person_box = None
    for obj in objs:
        bbox = obj.bbox
        x0, y0, x1, y1 = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
        bboxes.append([x0, y0, x1 - x0, y1 - y0])
        class_ids.append(obj.id)
        tag = configuration.labels.get(obj.id, obj.id)

        if tag == 'person':
            person_box = bbox
            cv2.rectangle(cv2_image, (x0, y0), (x1, y1), (255, 0, 255), 2)
            person_cor = (x0, y0, x1, y1)
            person_center = (int((person_cor[0] + person_cor[2]) / 2), int((person_cor[1] + person_cor[3]) / 2))
            cv2.circle(cv2_image, person_center, 10, (255, 0, 255), cv2.FILLED)

    # analyze connections with hands.
    hand_connections(objs, hands, bboxes, class_ids, possible_connections)

    # analyze connections with person.
    person_connections(person_box, bboxes, class_ids, possible_connections)

    return cv2_image, person_center, person_box


def initial():
    run_server()
    send_data(ActionsTypes.TURN_ON)
    accept_connection()


def main():
    # creating objects
    firebase = FirebaseConfig()

    # firebase configurations
    db = firebase.firebase
    categories = CategoriesList(firebase, db)
    configuration.add_categories(categories)
    first_person_show = False

    data_form = form()
    models_store = DetectorStore()

    while True:
        print("waiting for power on to get started")
        while not firebase.is_on():
            time.sleep(2)
            pass

        initial()
        user_name = firebase.get_name()

        print("capture started")
        firebase.reset_data()
        # TODO: robot should wake up and start to capture.
        move_direction = None

        while True:
            class_ids = []
            possible_connections = {}
            connections = []

            if firebase.is_changed():
                print("categories changed")
                categories.update_lists()

                firebase.update_finished()
                print("categories update")

            if firebase.is_report():
                print("producing report......")
                data_form.print_report(firebase)
                print("produce finished.")

            if (cv2.waitKey(1) & 0xFF == ord('q')) or (not firebase.is_on()):
                data_form.print_report(firebase)
                send_data(ActionsTypes.TURN_OFF)
                break

            is_person = False

            for i in range(configuration.number_of_iterations):
                bboxes = []
                frame = process_data()
                if frame is None:
                    continue

                cv2_im = frame.copy()

                threshold = 0.3
                objs = get_objects(frame, threshold, models_store.object_detector, cv2_im, )

                hands, cv2_im = models_store.hand_detector.get_model().findHands(cv2_im)
                cv2_im, person_center, person_box = append_objs_to_img(cv2_im, objs, class_ids, bboxes, hands,
                                                                       possible_connections)
                is_person = person_center is not None or is_person
                if first_person_show == False and person_center is not None:
                    first_person_show = True
                    st = user_name + " has entered the house."
                    events = [st]
                    data_form.print2file(events, firebase)
                    speak("hello" + user_name)

                if person_center is not None:
                    current_move = camera_move_check(cv2_im, configuration.moving_sensitivity, person_center,
                                                     person_box)
                    if current_move:
                        send_data(current_move)
                    move_direction = current_move
                else:
                    move_direction = None

                cv2.imshow("Image", cv2_im)
            for key in possible_connections.keys():
                connections.append(key)

            analyze_connections(connections, data_form, user_name, firebase, configuration.categories)
            if is_person and move_direction is None:
                send_data(ActionsTypes.STOP)
            elif move_direction is None and not is_person:
                send_data(ActionsTypes.START)

        print("capture stopped")


if __name__ == '__main__':
    main()
