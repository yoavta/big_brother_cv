import Lights
import time
from form import form
import argparse
import cv2
import os
from cvzone.HandTrackingModule import HandDetector
from subprocess import call
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from categories_list import categories_list
from rasbppery import Camera
from firebase_config import firebase_config

# initialize hand detector.
detector = HandDetector(detectionCon=0.6, maxHands=2)

# creating objects
firebase = firebase_config()
data_form = form()
my_camera = Camera(0, -10)

# firebase configurations
db = firebase.firebase.database()
categories = categories_list(firebase, db)
name = db.child("name").get().val()

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
default_model_dir = '../all_models'
default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
default_labels = 'coco_labels.txt'
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path',
                    default=os.path.join(default_model_dir, default_model))
parser.add_argument('--labels', help='label file path',
                    default=os.path.join(default_model_dir, default_labels))
parser.add_argument('--top_k', type=int, default=5,
                    help='number of categories with highest score to display')
parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=0)
parser.add_argument('--threshold', type=float, default=0.25,
                    help='classifier score threshold')
args = parser.parse_args()
print('Loading {} with {} labels.'.format(args.model, args.labels))
interpreter = make_interpreter(args.model)
interpreter.allocate_tensors()
labels = read_label_file(args.labels)
inference_size = input_size(interpreter)

# initialize openCV and Camera settings
cap = cv2.VideoCapture(0)
screen_width = 640
screen_height = 360
cap.set(3, screen_width)
cap.set(4, screen_height)


# speak from text function
def speak(text):
    cmd_beg = 'pico2wave -w testpico.wav "'
    cmd_end = '" && paplay testpico.wav'  # To play back the stored .wav file and to dump the std errors to /dev/null
    call([cmd_beg + text + cmd_end], shell=True)


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
                Lights.alarm_once(3)
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


# checkin if the camera should move to keep track the person. if it should move- execute moving.
def camera_move_check(rectangle_size, center_object):
    half_rec_size = int(rectangle_size / 2)
    shift = 40
    center_screen = (int(screen_width / 2), int(screen_height / 2) + shift)
    top_left = (center_screen[0] - half_rec_size, center_screen[1] - half_rec_size)

    cv2.rectangle(cv2_im, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)
    x, y = center_object[0], center_object[1]
    left_x = top_left[0]
    right_x = top_left[0] + rectangle_size
    lower_y = top_left[1]
    upper_y = top_left[1] + rectangle_size
    step = 2

    if x < left_x:
        my_camera.move_left(step)
    if x > right_x:
        my_camera.move_right(step)

    if y < lower_y:
        my_camera.move_up(step)

    if y > upper_y:
        my_camera.move_down(step)


# taking objects detected and mark them on the image.
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
    name = db.child("name").get().val()

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
            cv2_im = frame

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(interpreter, args.threshold)[:args.top_k]

            hands, cv2_im = detector.findHands(cv2_im)
            cv2_im, person_center = append_objs_to_img(cv2_im, inference_size, objs, labels)
            if first_person_show == False and person_center != None:
                first_person_show = True
                st = name + " has entered the house."
                events = []
                events.append(st)
                data_form.print2file(events, firebase)
                speak("hello" + name)

            moving_sensitivity = 140
            if person_center is not None:
                camera_move_check(moving_sensitivity, person_center)

            cv2.imshow("Image", cv2_im)

        confidence_level = 2
        for key in possible_connections.keys():
            if possible_connections.get(key) >= confidence_level:
                connections.append(key)

        analyze_connections(connections)

    print("capture stoped")
