import os

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


class DetectorStore:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.hand_detector = HandDetectionModel()


class ObjectDetector:
    def __init__(self):
        model, classes, colors, output_layers = load_yolo()
        self.object_detection_model = [model, classes, colors, output_layers]

    def get_model(self):
        return self.object_detection_model

    def detect_objects(self, img):
        model, classes, colors, output_layers = self.get_model()
        blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        outputs = model.forward(output_layers)
        return blob, outputs


class HandDetectionModel:
    def __init__(self):
        self.hand_detector_model = HandDetector(detectionCon=0.6, maxHands=2)

    def get_model(self):
        return self.hand_detector_model


def load_yolo():
    root_dir = os.path.abspath(os.path.dirname(__file__))

    weights_path = os.path.join(root_dir, "../Resources/yolo/yolov3.weights")
    cfg_path = os.path.join(root_dir, "../Resources/yolo/yolov3.cfg")
    names_path = os.path.join(root_dir, "../Resources/coco.names.txt")

    net = cv2.dnn.readNet(weights_path, cfg_path)

    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers
