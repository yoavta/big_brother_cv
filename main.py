import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from categories_list import CategoriesList
# from Camera import Camera
from firebase_config_example import FirebaseConfig
from form import form

# initialize hand detector.
from utils import speak

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

# Extracting object names:
classesFile = "Resources/coco.names.txt"
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
f.close()

# Extracting pearson triggers:
classesFile = "Resources/person.txt"
with open(classesFile, 'rt') as f:
    person_list = f.read().rstrip('\n').split('\n')
f.close()

# Extracting hand triggers:
classesFile = "Resources/hand.txt"
with open(classesFile, 'rt') as f:
    hand_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/other.txt"
with open(classesFile, 'rt') as f:
    other_list = f.read().rstrip('\n').split('\n')
f.close()

# initialize tensorflow object detector.
# default_model_dir = 'Resources/ssd_mobilenet_v2_2'
# default_model = 'ssd_mobilenet_v2.tflite'
default_labels = 'coco_labels.txt'

# initialize openCV and Camera settings
cap = cv2.VideoCapture(0)
screen_width = 640
screen_height = 360
cap.set(3, screen_width)
cap.set(4, screen_height)


# initialized model
# parser = argparse.ArgumentParser()
# parser.add_argument('--verbose', help="To print statements", default=True)
# args = parser.parse_args()


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


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
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
        list_in = categories.which_list_am_i_complet(con)
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
            # upload important images to firebase.
            #             count = str(data_form.important_num)
            #             file_name = "important"+count+".jpg"
            #             print(file_name)
            #             cv2.imwrite(file_name, cv2_im)
            #             firebase.upload_img(file_name)

    # printing to file
    data_form.print2file(events, firebase)


# analyze connections with hands. check if there is overlapping between objects and hands.
def hand_connections(objs):
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
            if labels.get(obj_id, obj_id) in hand_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    pass
                else:
                    if labels.get(obj_id, obj_id) in current_frame_connections:
                        continue
                    current_frame_connections.append(labels.get(obj_id, obj_id))
                    if labels.get(obj_id, obj_id) in possible_connections:
                        possible_connections[labels.get(obj_id, obj_id)] = \
                            possible_connections[labels.get(obj_id, obj_id)] + 1
                    else:
                        possible_connections[labels.get(obj_id, obj_id)] = 1


# analyze connections with person. check if there is overlapping between objects and person.
def person_connections(person_box):
    if person_box is not None:
        current_frame_connections = []
        x2, y2, w2, h2 = person_box[0], person_box[1], person_box[2], person_box[3]
        total_cons = len(bboxes)
        for j in range(total_cons):
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = classIds[j]
            if labels.get(obj_id, obj_id) in person_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    pass
                else:
                    if labels.get(obj_id, obj_id) in current_frame_connections:
                        continue
                    #                     print(labels.get(obj_id,obj_id))
                    current_frame_connections.append(labels.get(obj_id, obj_id))
                    if labels.get(obj_id, obj_id) in possible_connections:
                        possible_connections[labels.get(obj_id, obj_id)] = \
                            possible_connections[labels.get(obj_id, obj_id)] + 1
                    else:
                        possible_connections[labels.get(obj_id, obj_id)] = 1


def append_objs_to_img(cv2_image, inference_size, objs, labels):
    height, width, channels = cv2_image.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    person_center = None
    person_box = None
    for obj in objs:
        percent = int(100 * obj.score)
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        bboxes.append([x0, y0, x1 - x0, y1 - y0])
        classIds.append(obj.id)

        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        tag = labels.get(obj.id, obj.id)

        cv2_image = cv2.rectangle(cv2_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_image = cv2.putText(cv2_image, label, (x0, y0 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # in case object is person.
        if tag == 'person':
            person_box = bbox
            person_cor = (x0, y0, x1, y1)
            person_center = (int((person_cor[0] + person_cor[2]) / 2), int((person_cor[1] + person_cor[3]) / 2))
            cv2.circle(cv2_image, person_center, 3, (220, 20, 20), 2)

    # analyze connections with hands.
    hand_connections(objs)

    # analyze connections with person.
    person_connections(person_box)

    return cv2_image, person_center


while True:
    print("waiting for power on to get started")
    while not firebase.is_on():
        time.sleep(2)
        pass
    name = db.reference("name").get()

    print("capture started")
    firebase.initial()
    while True:
        classIds = []
        confs = []
        possible_connections = {}
        connections = []

        if firebase.is_changed():
            print(categories.get_importants())
            print("categories changed")
            categories.update_lists()

            print(categories.get_importants())
            firebase.update_finished()
            print("categories update")

        if firebase.is_report():
            print("producing report......")
            data_form.print_report(firebase)
            print("produce finished.")

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (not firebase.is_on()):
            data_form.print_report(firebase)
            break

        for i in range(4):
            bboxes = []
            ret, frame = cap.read()
            if not ret:
                break
            height, width, channels = frame.shape
            model, classes, colors, output_layers = load_yolo()
            blob, outputs = detect_objects(frame, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

            # objs = get_objects(threshold)
            # objs = objs[:args.top_k]
            #
            # hands, cv2_im = detector.findHands(cv2_im)
            # cv2_im, person_center = append_objs_to_img(cv2_im, inference_size, objs, labels)
            # if first_person_show == False and person_center != None:
            #     first_person_show = True
            #     st = name + " has entered the house."
            #     events = []
            #     events.append(st)
            #     data_form.print2file(events, firebase)
            #     speak("hello" + name)
            #
            # moving_sensitivity = 140
            # # if person_center is not None:
            # #     camera_move_check(moving_sensitivity, person_center)
            #
            # cv2.imshow("Image", cv2_im)

        confidence_level = 2
        for key in possible_connections.keys():
            if possible_connections.get(key) >= confidence_level:
                connections.append(key)

        analyze_connections(connections)

    print("capture stopped")
