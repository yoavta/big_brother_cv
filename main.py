import math
import cv2
import numpy as np
import Lights
import time
from form import form
from cvzone.HandTrackingModule import HandDetector
from rasbppery import Camera

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
detector = HandDetector(detectionCon = 0.4, maxHands = 2)
data_form = form()
# my_camera = Camera(-0.3,0)

#define Name

name = 'yoav'

# Extracting object names:
classesFile = "Resources/coco.names.txt"
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/sitting.txt"
with open(classesFile, 'rt') as f:
    sitting_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/danger.txt"
with open(classesFile, 'rt') as f:
    danger_list = f.read().rstrip('\n').split('\n')
f.close()

classesFile = "Resources/food.txt"
with open(classesFile, 'rt') as f:
    food_list = f.read().rstrip('\n').split('\n')
f.close()

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

# Connecting to the network:
model_configuration = 'yolov3-tiny.cfg'
model_weights = 'yolov3-tiny.weights'
net = cv2.dnn.readNetFromDarknet(model_configuration,model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# print(classes_names)


def analyze_connections(connections):
    events =[]
    for con in connections:
        if con in computer_list:
            st = name + " using the computer with " + con + "."
        elif con in danger_list:
            # Lights.alarm_once(5)
            st = "watch out!! " + name + " is playing with " + con + "."
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
    data_form.print2file(events)


def hand_connections(indices):
    for hand in hands:
        x2, y2, w2, h2 = hand['bbox'][0], hand['bbox'][1], hand['bbox'][2], hand['bbox'][3]
        for j in indices:
            j = j[0]
            box = bbox[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            if classNames[classIds[j]] in hand_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    # print('No overlapping')
                    pass
                else:
                    if classNames[classIds[j]] in possible_connections:
                        possible_connections[classNames[classIds[j]]] =\
                            possible_connections[classNames[classIds[j]]] + 1
                    else:
                        possible_connections[classNames[classIds[j]]] = 1


def person_connections(person_box, indices):
    if person_box is not None:
        x2, y2, w2, h2 = person_box[0], person_box[1], person_box[2], person_box[3]
        for j in indices:
            j = j[0]
            box = bbox[j]
            x1, y1, w1, h1, = box[0], box[1], box[2], box[3]
            if classNames[classIds[j]] in person_list:
                if (x1 >= x2 + w2) or (x2 >= x1 + w1) or (y1 >= y2 + h2) or (y2 >= y1 + h1):
                    # print('No overlapping')
                    pass
                else:
                    if classNames[classIds[j]] in possible_connections:
                        possible_connections[classNames[classIds[j]]] =\
                            possible_connections[classNames[classIds[j]]] + 1
                    else:
                        possible_connections[classNames[classIds[j]]] = 1


def find_objects(outputs, image):
    h_target, w_target, c_target = img.shape

    for output in outputs:
        for det in output:
            # Get highest probability
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Filter inaccurate findings
            if confidence > confThreshold:
                w, h = int(det[2] * w_target), int(det[3] * h_target)  # Taking the width and height of the box
                x, y = int((det[0] * w_target) - (w / 2)), int((det[1] * h_target) - (h / 2))
                bbox.append([x, y, w, h])
                classIds.append(class_id)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    person_box=None
    person_center = None
    for k in indices:
        k = k[0]
        box = bbox[k]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[k]].upper()} {int(confs[k] * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # print(str(i) + ':')
        # print(classNames[classIds[i]], "{:.2f}".format(confs[i]))
        if classNames[classIds[k]] == 'person':
            person_box = box
            person_cor = (x, y, (x + w), (y + h))
            person_center = (int((person_cor[0]+person_cor[2])/2), int((person_cor[1]+person_cor[3])/2))
            cv2.circle(img, person_center, 3, (220, 20, 20), 2)

    person_connections(person_box, indices)
    hand_connections(indices)
    return person_center




def camera_move_check(rectangle_size, center_object):
    half_rec_size = int(rectangle_size/2)
    center_screen = (int(screen_width/2), int(screen_height/2)+100)
    top_left= (center_screen[0]-half_rec_size,center_screen[1]-half_rec_size)

    cv2.rectangle(img, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)
    x,y = center_object[0],center_object[1]
    left_x= top_left[0]
    right_x= top_left[0]+rectangle_size
    lower_y= top_left[1]
    upper_y=top_left[1]+rectangle_size


    if x < left_x:
        # print("camera moving left")
        my_camera.move_pan_one_step(1)

    if x> right_x:
        # print("camera moving right")
        my_camera.move_pan_one_step(-1)

    if y< lower_y:
        # print("camera moving up")
        my_camera.move_tilt_one_step(-1)

    if y> upper_y:
        # print("camera moving down")
        my_camera.move_tilt_one_step(1)

cap = cv2.VideoCapture(0)
screen_width = 1000
screen_height = 500
cap.set(3, screen_width)
cap.set(4, screen_height)


while True:
    bbox = []
    classIds = []
    confs = []
    possible_connections = {}
    connections = []
    if cv2.waitKey(1) & 0xFF == ord('q'):
        data_form.print_report()
        break
    for i in range(4):
        success, img = cap.read()  # Read camera
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)  # Make blob for net
        net.setInput(blob)  # Input blob
        layerNames = net.getLayerNames()  # Find the names of the three layers
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # Get the names of the layers
        outputs = net.forward(outputNames)  # The data of thr layers: (Number of boxes, 85 values: x, y, width, height,
                                            # confidence, and 80 predictions for which object it is
        hands, img = detector.findHands(img)
        person_center = find_objects(outputs, img)
        moving_sensitivity= 150

        # if person_center is not None:
        #     camera_move_check(moving_sensitivity, person_center)

        cv2.imshow("Image", img)



    confidence_level = 3
    for key in possible_connections.keys():
        # print(key + possible_connections.get(key)+"")
        if possible_connections[key] >= confidence_level:
            connections.append(key)

    # print(connections)
    analyze_connections(connections)






