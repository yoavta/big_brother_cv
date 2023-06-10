from typing import List

import cv2

from src.Object import Object


def draw_labels(objects: List[Object], colors, img,classes):
    font = cv2.FONT_HERSHEY_PLAIN
    for obj in objects:
        x, y, w, h = obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax
        label = str(classes[obj.id])
        color = colors[obj.id]
        cv2.rectangle(img, (x, y), (w, h), color, 2)
        cv2.putText(img, label, (x, y + 15), font, 1, color, 1)
    cv2.imshow("Image", img)
