import os

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from src.Bbox import get_box_dimensions, BBox
from src.Object import Object
from src.configuration import Configuration
from src.image_utils import draw_labels


class DetectorStore:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.hand_detector = HandDetectionModel()

    def append_objs_to_img(self, cv2_image, objs, class_ids, bboxes, hands, possible_connections,
                           configuration=Configuration()):
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
        self.hand_detector.hand_connections(objs, hands, bboxes, class_ids, possible_connections)

        # analyze connections with person.
        self.object_detector.person_connections(person_box, bboxes, class_ids, possible_connections)

        return cv2_image, person_center, person_box


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

    def get_objects(self,frame, threshold, object_detector, img):
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

    # analyze connections with person. check if there is overlapping between objects and person.
    def person_connections(self,person_box: BBox, bboxes, class_ids, possible_connections, configuration=Configuration()):
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


class HandDetectionModel:
    def __init__(self):
        self.hand_detector_model = HandDetector(detectionCon=0.6, maxHands=2)

    def get_model(self):
        return self.hand_detector_model

    # analyze connections with hands. check if there is overlapping between objects and hands.
    def hand_connections(self,objs, hands, bboxes, class_ids, possible_connections, configuration=Configuration()):
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
