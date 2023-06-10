import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from categories_list import CategoriesList
from configuration import Configuration
from controller_client import ActionsTypes
from controller_client import send_data
from controller_server import run_server, accept_connection, process_data
# from Camera import Camera
from firebase_config_example import FirebaseConfig
from form import form
# initialize hand detector.
from utils import speak, read_label_file

detector = HandDetector(detectionCon=0.6, maxHands=2)

# creating objects
firebase = FirebaseConfig()

data_form = form()


# my_camera = Camera(0, -10)

class Object:
    def __init__(self, class_id, score, bbox):
        self.id = class_id
        self.score = score
        self.bbox = bbox


class BBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


# firebase configurations
db = firebase.firebase
categories = CategoriesList(firebase, db)
name = db.reference("name").get()

first_person_show = False


configuration = Configuration()


def start_webcam():
    cap = cv2.VideoCapture(0)

    return cap


def display_blob(blob):
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, colors, class_ids, classes, img, indexes):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def load_yolo():
    net = cv2.dnn.readNet("Resources/yolo/yolov3.weights", "Resources/yolo/yolov3.cfg")
    classes = []
    with open("Resources/coco.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


# linking object connection to a categories and produce update text.
def analyze_connections(connections):
    events = []
    for con in connections:
        list_in = categories.which_list_am_i_complete(con)
        if list_in == "computer":
            st = name + " is using the computer with a(n) " + con + "."
        elif list_in == "danger":
            st = name + " is playing with a " + con + "."
            if con in categories.get_importants():
                st = "watch out!! " + name + " is playing with a(n) " + con + "."
                # speaking and open warning lights.
                speak(st)
                # Lights.alarm_once(3)
        elif list_in == "food":
            st = name + " is eating a(n) " + con + "."
        elif list_in == "holdings":
            st = name + " is holding a(n) " + con + "."
        elif list_in == "playing":
            st = name + " is playing with a(n) " + con + "."
        elif list_in == "sitting":
            st = name + " is sitting on a(n) " + con + "."
        elif list_in == "specific":
            if con == 'toothbrush':
                st = name + "is brushing her teeth."
            elif con == 'book':
                st = name + " is reading a book."
            elif con == 'cell phone':
                st = name + " is using her phone."
        elif list_in == "tv":
            st = name + " is using the tv with a(n)" + con + "."
        elif list_in == "wearing":
            st = name + " is wearing a(n) " + con + "."

        else:
            st = name + " has a connection with a(n) " + con + "."
        events.append(st)

        # check if this connection eas nark as important.
        if con in categories.get_importants():
            data_form.add_important(st)

    # printing to file
    data_form.print2file(events, firebase)

    return events


# analyze connections with hands. check if there is overlapping between objects and hands.
def hand_connections(objs, hands, bboxes, classIds, possible_connections):
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
            obj_id = classIds[j]
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
def person_connections(person_box: BBox, bboxes, classIds, possible_connections):
    if person_box is not None:
        current_frame_connections = []
        x2, y2, w2, h2 = [person_box.xmin, person_box.ymin, person_box.xmax, person_box.ymax]
        total_cons = len(bboxes)
        for j in range(total_cons):
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = classIds[j]
            if configuration.labels.get(obj_id, obj_id) in configuration.person_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    pass
                else:
                    if configuration.labels.get(obj_id, obj_id) in current_frame_connections:
                        continue
                    #                     print(labels.get(obj_id,obj_id))
                    current_frame_connections.append(configuration.labels.get(obj_id, obj_id))
                    if configuration.labels.get(obj_id, obj_id) in possible_connections:
                        possible_connections[configuration.labels.get(obj_id, obj_id)] = \
                            possible_connections[configuration.labels.get(obj_id, obj_id)] + 1
                    else:
                        possible_connections[configuration.labels.get(obj_id, obj_id)] = 1


def get_objects(frame, threshold, model_loaded, img):
    height, width, channels = frame.shape
    model, classes, colors, output_layers = model_loaded
    blob, outputs = detect_objects(frame, model, output_layers)
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


def camera_move_check(img, rectangle_size, center_object, person_box: BBox) -> ActionsTypes:
    half_rec_size = int(rectangle_size / 2)
    center_screen = (int(configuration.screen_width / 2), int(configuration.screen_height / 2))
    top_left = (center_screen[0] - half_rec_size, center_screen[1] - half_rec_size)

    cv2.rectangle(img, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)
    x, y = center_object[0], center_object[1]
    left_x = top_left[0]
    right_x = top_left[0] + rectangle_size

    if x < left_x:
        return ActionsTypes.LEFT

    if x > right_x:
        return ActionsTypes.RIGHT


def append_objs_to_img(cv2_image, objs, classIds, bboxes, hands, possible_connections):
    person_center = None
    person_box = None
    for obj in objs:
        bbox = obj.bbox
        x0, y0, x1, y1 = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
        bboxes.append([x0, y0, x1 - x0, y1 - y0])
        classIds.append(obj.id)
        tag = configuration.labels.get(obj.id, obj.id)

        if tag == 'person':
            person_box = bbox
            cv2.rectangle(cv2_image, (x0, y0), (x1, y1), (255, 0, 255), 2)
            person_cor = (x0, y0, x1, y1)
            person_center = (int((person_cor[0] + person_cor[2]) / 2), int((person_cor[1] + person_cor[3]) / 2))
            cv2.circle(cv2_image, person_center, 10, (255, 0, 255), cv2.FILLED)

    # analyze connections with hands.
    hand_connections(objs, hands, bboxes, classIds, possible_connections)

    # analyze connections with person.
    person_connections(person_box, bboxes, classIds, possible_connections)

    return cv2_image, person_center, person_box


def main():
    model, classes, colors, output_layers = load_yolo()
    model_loaded = [model, classes, colors, output_layers]
    global first_person_show, move_direction
    print("waiting for power on to get started")
    while not firebase.is_on():
        time.sleep(2)
        pass
    run_server()
    send_data(ActionsTypes.TURN_ON)
    accept_connection()

    name = db.reference("name").get()

    print("capture started")
    firebase.initial()
    # TODO: robot should wake up and start to capture.

    while True:
        classIds = []
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
            break

        is_person = False

        for i in range(configuration.NUMBER_OF_ITERATIONS):
            bboxes = []
            frame = process_data()
            if frame is None:
                continue

            cv2_im = frame.copy()

            threshold = 0.3
            objs = get_objects(frame, threshold, model_loaded, cv2_im, )

            hands, cv2_im = detector.findHands(cv2_im)
            cv2_im, person_center, person_box = append_objs_to_img(cv2_im, objs, classIds, bboxes, hands,
                                                                   possible_connections)
            is_person = person_center is not None or is_person
            if first_person_show == False and person_center != None:
                first_person_show = True
                st = name + " has entered the house."
                events = []
                events.append(st)
                data_form.print2file(events, firebase)
                speak("hello" + name)
            #
            moving_sensitivity = 220
            move_direction = None
            if person_center is not None:
                move_direction = camera_move_check(cv2_im, moving_sensitivity, person_center, person_box)
                if move_direction:
                    send_data(move_direction)

            cv2.imshow("Image", cv2_im)
        for key in possible_connections.keys():
            connections.append(key)

        results = analyze_connections(connections)
        if is_person and move_direction is None:
            send_data(ActionsTypes.STOP)
        elif move_direction is None and not is_person:
            send_data(ActionsTypes.START)

    print("capture stopped")


if __name__ == '__main__':
    main()
