import math
import cv2
import numpy as np
import Lights
import time
from form import form
from cvzone.HandTrackingModule import HandDetector
from Camera import Camera
from firebase_config import firebase_config
import pyrebase
from num2words import num2words
from subprocess import call
import argparse
import cv2
import os
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from categories_list import categories_list


detector = HandDetector(detectionCon = 0.6, maxHands = 2)
data_form = form()
my_camera = Camera(30,30)
firebase = firebase_config()
#define Name
db = firebase.firebase.database()
categories = categories_list(firebase,db)

name = db.child("name").get().val()
# Extracting object names:
classesFile = "Resources/coco.names.txt"
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/person.txt"
with open(classesFile, 'rt') as f:
    person_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/hand.txt"
with open(classesFile, 'rt') as f:
    hand_list = f.read().rstrip('\n').split('\n')
f.close()


classesFile = "Resources/other.txt"
with open(classesFile, 'rt') as f:
    other_list = f.read().rstrip('\n').split('\n')
f.close()



#################



default_model_dir = '../all_models'
default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
default_labels = 'coco_labels.txt'
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path',
                    default=os.path.join(default_model_dir,default_model))
parser.add_argument('--labels', help='label file path',
                    default=os.path.join(default_model_dir, default_labels))
parser.add_argument('--top_k', type=int, default=8,
                    help='number of categories with highest score to display')
parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
parser.add_argument('--threshold', type=float, default=0.35,
                    help='classifier score threshold')
args = parser.parse_args()

print('Loading {} with {} labels.'.format(args.model, args.labels))
interpreter = make_interpreter(args.model)
interpreter.allocate_tensors()
labels = read_label_file(args.labels)
inference_size = input_size(interpreter)

cap = cv2.VideoCapture(0)
screen_width = 1000
screen_height = 500
cap.set(3, screen_width)
cap.set(4, screen_height)



#############



def speek(text):

    
    cmd_beg= 'pico2wave -w testpico.wav "'
    cmd_end= '" && paplay testpico.wav' # To play back the stored .wav file and to dump the std errors to /dev/null

    call([cmd_beg+text+cmd_end], shell=True)



def analyze_connections(connections):
    events =[]
    for con in connections:
        list_in = categories.which_list_am_i_complet(con)
        if list_in == "computer":
            st = name + " using the computer with " + con + "."
        elif list_in == "danger":
            st = name + " is playing with " + con + "."
            if con in categories.get_importants():
                st = "watch out!! " + name + " is playing with " + con + "."
                speek(st)
                Lights.alarm_once(3)
        elif list_in == "food":
            st = name + " is  eating a " + con + "."
        elif list_in == "holdings":
            st = name + " is  holding a " + con + "."
        elif list_in == "playing":
            st = name + " is playing with "+ con + "."
        elif list_in == "sitting":
            st = name + " is sitting on a "+ con + "."
        elif list_in == "specific":
            if con == 'toothbrush':
                st = name + "is brushing is teeth."
            elif con == 'book':
                st = name + " is reading a book."
            elif con == 'cell phone':
                st = name + " using his phone."
        elif list_in  == "tv":
            st = name + " using the tv with " + con + "."
        elif list_in == "wearing":
            st = name + " is wearing a " + con + "."
        else:
            st = name + " has connection with " + con + "."
        # print(st)
        events.append(st)

        if con in categories.get_importants():
            data_form.add_important(st)
            
    data_form.print2file(events,firebase)


def hand_connections(objs):
    for hand in hands:
        current_frame_connections = []

        x2, y2, w2, h2 = hand['bbox'][0], hand['bbox'][1], hand['bbox'][2], hand['bbox'][3]
        total_cons= len(bboxes)
        last_cons = len(objs)
        if last_cons == 0 or total_cons == 0:
            break
        index_start_from = total_cons-last_cons
        for j in range(total_cons):
#             j=index_start_from
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = classIds[j]
#             print(labels.get(obj_id,obj_id))
#             if labels.get(obj_id,obj_id) == "bottle":
#                 print("this")
            if labels.get(obj_id,obj_id) in hand_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    # print('No overlapping')
                    pass
                else:
                    if labels.get(obj_id,obj_id) in current_frame_connections:
                        continue
#                     print(labels.get(obj_id,obj_id))
                    current_frame_connections.append(labels.get(obj_id,obj_id))
                    if labels.get(obj_id,obj_id) in possible_connections:
                        possible_connections[labels.get(obj_id,obj_id)] =\
                            possible_connections[labels.get(obj_id,obj_id)] + 1
                    else:
                        possible_connections[labels.get(obj_id,obj_id)] = 1


def person_connections(person_box,objs):
    if person_box is not None:
        current_frame_connections = []
        x2, y2, w2, h2 = person_box[0], person_box[1], person_box[2], person_box[3]
        total_cons= len(bboxes)
        last_cons = len(objs)
        index_start_from = total_cons-last_cons
        for j in range(total_cons):
#             j=index_start_from
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = classIds[j]
            if labels.get(obj_id,obj_id) in person_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    # print('No overlapping')
                    pass
                else:
                    if labels.get(obj_id,obj_id) in current_frame_connections:
                        continue
#                     print(labels.get(obj_id,obj_id))
                    current_frame_connections.append(labels.get(obj_id,obj_id))
                    if labels.get(obj_id,obj_id) in possible_connections:
                        possible_connections[labels.get(obj_id,obj_id)] =\
                            possible_connections[labels.get(obj_id,obj_id)] + 1
                    else:
                        possible_connections[labels.get(obj_id,obj_id)] = 1





def camera_move_check(rectangle_size, center_object):
    half_rec_size = int(rectangle_size/2)
    shift = 0
    center_screen = (int(screen_width/2), int(screen_height/2)+shift)
    top_left= (center_screen[0]-half_rec_size,center_screen[1]-half_rec_size)

    cv2.rectangle(cv2_im, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)
    x,y = center_object[0],center_object[1]
    left_x= top_left[0]
    right_x= top_left[0]+rectangle_size
    lower_y= top_left[1]
    upper_y=top_left[1]+rectangle_size
    step = 2


    if x < left_x:
        #print("camera moving left")
        #print(my_camera.move_left(1))
        my_camera.move_left(step)
    if x> right_x:
        #print("camera moving right")
        #print(my_camera.move_right(1))
        my_camera.move_right(step)

    if y< lower_y:
        #print("camera moving up")
        #print(my_camera.move_up(1))
        my_camera.move_up(step)

    if y> upper_y:
        #print("camera moving down")
        #print(my_camera.move_down(1))
        my_camera.move_down(step)


####################


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    person_center= None
    person_box = None
    for obj in objs:
        percent = int(100 * obj.score)
#         if percent < 30:
#             objs.remove(obj)
#             continue
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        bboxes.append([x0, y0, x1-x0, y1-y0])
        classIds.append(obj.id)

        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        tag = labels.get(obj.id, obj.id)
        
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        if  tag == 'person':
            person_box = bbox
            person_cor = (x0, y0, x1,y1)
            person_center = (int((person_cor[0] + person_cor[2]) / 2), int((person_cor[1] + person_cor[3]) / 2))
            cv2.circle(cv2_im, person_center, 3, (220, 20, 20), 2)
#     print("before: " + str(possible_connections))
    hand_connections(objs)
#     print("hand: " + str(possible_connections))
    person_connections(person_box,objs)
#     print("person: " + str(possible_connections))

    
    
    return cv2_im, person_center
######################



while True:
#     for l in categories_list:
#         firebase.tick_categories(l , 0)

    print("waiting for power on to get started")
    while not firebase.is_on():
        time.sleep(2)
        pass
    
    print("capture started")
    firebase.initial()
    while True:
    #     bboxes = []
        classIds = []
        confs = []
        possible_connections = {}
        connections = []
        
        if firebase.is_changed():
#             print(computer_list)
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
    #         print("i :"+ str(i))
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
            
            moving_sensitivity= 200
            if person_center is not None:
                camera_move_check(moving_sensitivity, person_center)

            cv2.imshow("Image", cv2_im)



        confidence_level = 2
        for key in possible_connections.keys():
            if possible_connections.get(key) >= confidence_level:
                
                connections.append(key)
                

        analyze_connections(connections)

    print("capture stoped")







