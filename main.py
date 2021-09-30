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
counter = 1
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
detector = HandDetector(detectionCon = 0.6, maxHands = 2)
data_form = form()
my_camera = Camera(0,-20)

firebase = firebase_config()

#define Name
db = firebase.firebase.database()
name = db.child("name").get().val()

# Extracting object names:
classesFile = "Resources/coco.names.txt"
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/sitting.txt"
with open(classesFile, 'rt') as f:
    sitting_list = f.read().rstrip('\n').split('\n')
f.close()


danger_list = firebase.update(db.child("data").child("categories").child("danger").get().val())
food_list = firebase.update(db.child("data").child("categories").child("food").get().val())

classesFile = "Resources/person.txt"
with open(classesFile, 'rt') as f:
    person_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/hand.txt"
with open(classesFile, 'rt') as f:
    hand_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/computer.txt"
with open(classesFile, 'rt') as f:
    computer_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/holdings.txt"
with open(classesFile, 'rt') as f:
    holdings_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/other.txt"
with open(classesFile, 'rt') as f:
    other_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/playing.txt"
with open(classesFile, 'rt') as f:
    playing_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/specific.txt"
with open(classesFile, 'rt') as f:
    specific_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/tv.txt"
with open(classesFile, 'rt') as f:
    tv_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/using.txt"
with open(classesFile, 'rt') as f:
    using_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/wearing.txt"
with open(classesFile, 'rt') as f:
    wearing_list = f.read().rstrip('\n').split('\n')
f.close()




#################
import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


default_model_dir = '../all_models'
default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
default_labels = 'coco_labels.txt'
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path',
                    default=os.path.join(default_model_dir,default_model))
parser.add_argument('--labels', help='label file path',
                    default=os.path.join(default_model_dir, default_labels))
parser.add_argument('--top_k', type=int, default=3,
                    help='number of categories with highest score to display')
parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
parser.add_argument('--threshold', type=float, default=0.1,
                    help='classifier score threshold')
args = parser.parse_args()

print('Loading {} with {} labels.'.format(args.model, args.labels))
interpreter = make_interpreter(args.model)
interpreter.allocate_tensors()
labels = read_label_file(args.labels)
inference_size = input_size(interpreter)

#cap = cv2.VideoCapture(args.camera_idx)


################





























# Connecting to the network:
model_configuration = 'yolov3-tiny.cfg'
model_weights = 'yolov3-tiny.weights'
net = cv2.dnn.readNetFromDarknet(model_configuration,model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# print(classes_names)



def speek(text):
    cmd_beg= 'espeak -ven+f4 '
    cmd_end= ' | aplay /home/pi/Desktop/Text.wav  2>/dev/null' # To play back the stored .wav file and to dump the std errors to /dev/null

    #Replacing ' ' with '_' to identify words in the text entered
    text = text.replace(' ', '_')

    #Calls the Espeak TTS Engine to read aloud a Text
    call([cmd_beg+text+cmd_end], shell=True)

def analyze_connections(connections):
    events =[]
    for con in connections:
        if con in computer_list:
            st = name + " using the computer with " + con + "."
        elif con in danger_list:
            st = "watch out!! " + name + " is playing with " + con + "."
            speek(st)
            Lights.alarm_once(3)
        elif con in food_list:
            st = name + " is  eating a " + con + "."
        elif con in holdings_list:
            st = name + " is  holding a " + con + "."
        elif con in playing_list:
            st = name + " is playing with "+ con + "."
        elif con in sitting_list:
            st = name + " is sitting on a "+ con + "."
        elif con in specific_list:
            if con == 'toothbrush':
                st = name + "is brushing is teeth."
            elif con == 'book':
                st = name + " is reading a book."
        elif con in tv_list:
            st = name + " using the tv with " + con + "."
        elif con in wearing_list:
            st = name + " is wearing a " + con + "."
        else:
            st = name + " has connection with " + con + "."
        # print(st)
        events.append(st)
    data_form.print2file(events,firebase)


def hand_connections(objs):
    for hand in hands:
        x2, y2, w2, h2 = hand['bbox'][0], hand['bbox'][1], hand['bbox'][2], hand['bbox'][3]
        total_cons= len(bboxes)
        last_cons = len(objs)
        if last_cons == 0 or total_cons == 0:
            break
        index_start_from = total_cons-last_cons
        for j in range(last_cons):
            j=index_start_from
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = classIds[j]
            if labels.get(obj_id,obj_id) in hand_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    # print('No overlapping')
                    pass
                else:
                    if labels.get(obj_id,obj_id) in possible_connections:
                        possible_connections[labels.get(obj_id,obj_id)] =\
                            possible_connections[labels.get(obj_id,obj_id)] + 1
                    else:
                        possible_connections[labels.get(obj_id,obj_id)] = 1


def person_connections(person_box,objs):
    if person_box is not None:
        x2, y2, w2, h2 = person_box[0], person_box[1], person_box[2], person_box[3]
        total_cons= len(bboxes)
        last_cons = len(objs)
        index_start_from = total_cons-last_cons
        for j in range(last_cons):
            j=index_start_from
            box = bboxes[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            obj_id = classIds[j]
            if labels.get(obj_id,obj_id) in person_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    # print('No overlapping')
                    pass
                else:
                    if labels.get(obj_id,obj_id) in possible_connections:
                        possible_connections[labels.get(obj_id,obj_id)] =\
                            possible_connections[labels.get(obj_id,obj_id)] + 1
                    else:
                        possible_connections[labels.get(obj_id,obj_id)] = 1

#         for j in indices:
#             j = j.id
#             print(j)
#             box = bboxes[j]
#             x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
#             print(classNames)
#             print(classIds)
#             if classNames[classIds[j]] in person_list:
#                 if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
#                     # print('No overlapping')
#                     pass
#                 else:
#                     if classNames[classIds[j]] in possible_connections:
#                         possible_connections[classNames[classIds[j]]] =\
#                             possible_connections[classNames[classIds[j]]] + 1
#                     else:
#                         possible_connections[classNames[classIds[j]]] = 1


# def find_objects(outputs, image):
#     h_target, w_target, c_target = img.shape
# 
#     for output in outputs:
#         for det in output:
#             # Get highest probability
#             scores = det[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             # Filter inaccurate findings
#             if confidence > confThreshold:
#                 w, h = int(det[2] * w_target), int(det[3] * h_target)  # Taking the width and height of the box
#                 x, y = int((det[0] * w_target) - (w / 2)), int((det[1] * h_target) - (h / 2))
#                 bbox.append([x, y, w, h])
#                 classIds.append(class_id)
#                 confs.append(float(confidence))
# 
#     indices = cv2.dnn.NMSBoxes(bboxes, confs, confThreshold, nmsThreshold)
#     print("indices:")
#     print(indices)
#     print("bboxes:")
#     print(bboxes)
#     print("confs:")
#     print(confs)
#     person_box=None
#     person_center = None
#     for k in indices:
#         k = k[0]
#         box = bboxes[k]
#         x, y, w, h = box[0], box[1], box[2], box[3]
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
#         cv2.putText(img, f'{classNames[classIds[k]].upper()} {int(confs[k] * 100)}%', (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
#         # print(str(i) + ':')
#         # print(classNames[classIds[i]], "{:.2f}".format(confs[i]))
#         if classNames[classIds[k]] == 'person':
#             person_box = box
#             person_cor = (x, y, (x + w), (y + h))
#             person_center = (int((person_cor[0]+person_cor[2])/2), int((person_cor[1]+person_cor[3])/2))
#             cv2.circle(img, person_center, 3, (220, 20, 20), 2)
#     hand_connections()
#     person_connections(person_box)
#     return person_center




def camera_move_check(rectangle_size, center_object):
    half_rec_size = int(rectangle_size/2)
    center_screen = (int(screen_width/2), int(screen_height/2)+100)
    top_left= (center_screen[0]-half_rec_size,center_screen[1]-half_rec_size)

#     cv2.rectangle(cv2_img, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)
    x,y = center_object[0],center_object[1]
    left_x= top_left[0]
    right_x= top_left[0]+rectangle_size
    lower_y= top_left[1]
    upper_y=top_left[1]+rectangle_size


    if x < left_x:
        # print("camera moving left")
        my_camera.move_left(1)

    if x> right_x:
        # print("camera moving right")
        my_camera.move_right(1)

    if y< lower_y:
        # print("camera moving up")
        my_camera.move_up(1)

    if y> upper_y:
        # print("camera moving down")
        my_camera.move_down(1)

cap = cv2.VideoCapture(0)
screen_width = 1000
screen_height = 500
cap.set(3, screen_width)
cap.set(4, screen_height)


####################


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    person_center= None
    person_box = None
    for obj in objs:
        percent = int(100 * obj.score)
        if percent < 40:
            objs.remove(obj)
            continue
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

    person_connections(person_box,objs)
    hand_connections(objs)

    
    
    return cv2_im, person_center
######################
while True:
    bboxes = []
    classIds = []
    confs = []
    possible_connections = {}
    connections = []

    if firebase.is_changed():
        print("categories changed")
        danger_list = firebase.update(db.child("data").child("categories").child("danger").get().val())
        food_list = firebase.update(db.child("data").child("categories").child("food").get().val())
        firebase.update_finished()
        print("categories update")

    for i in range(5):
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
        
#         success, img = cap.read()  # Read camera
        
        
        
        
    

#         blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)  # Make blob for net
#         net.setInput(blob)  # Input blob
#         layerNames = net.getLayerNames()  # Find the names of the three layers
#         outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # Get the names of the layers
#         outputs = net.forward(outputNames)  # The data of thr layers: (Number of boxes, 85 values: x, y, width, height,
#                                             # confidence, and 80 predictions for which object it is

#        person_center = find_objects(outputs, img)


        moving_sensitivity= 150
        if person_center is not None:
            camera_move_check(moving_sensitivity, person_center)

        cv2.imshow("Image", cv2_im)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or counter ==30:
            data_form.print_report(firebase)
            break

#     print("possible connections: "+ str(possible_connections)+"("+str(len(possible_connections))+")")

    confidence_level = 2
    for key in possible_connections.keys():
        # print(key + possible_connections.get(key)+"")
        if possible_connections[key] >= confidence_level:
            connections.append(key)
            
#     print("connections: "+ str(connections)+"("+str(len(connections))+")")

    # print(connections)
    analyze_connections(connections)
    counter = counter + 1






