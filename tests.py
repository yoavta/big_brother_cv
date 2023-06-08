from main import load_yolo, get_objects, detector,append_objs_to_img
import cv2

model, classes, colors, output_layers = load_yolo()
model_loaded = [model, classes, colors, output_layers]
frame = cv2.imread('output2.jpg')
frame = cv2.resize(frame, (640, 480))
cv2_im = frame.copy()
inference_size = (320, 320)

threshold = 0.6
bboxes = []
classIds = []

objs = get_objects(frame, threshold, model_loaded, cv2_im)
# show image
hands, cv2_im = detector.findHands(cv2_im)
moving_sensitivity = 140
rectangle_size = moving_sensitivity
screen_width = 640
screen_height = 480

half_rec_size = int(rectangle_size / 2)
center_screen = (int(screen_width / 2), int(screen_height / 2))
cv2.circle(cv2_im, center_screen, 2, (0, 0, 255), cv2.FILLED)
top_left = (center_screen[0] - half_rec_size, center_screen[1] - half_rec_size)

# cv2.rectangle(cv2_im, top_left, (top_left[0] + rectangle_size, top_left[1] + rectangle_size), (0, 0, 255), 2)


append_objs_to_img(cv2_im, objs)

cv2.imshow("Image", cv2_im)
cv2.waitKey(100)
cv2.destroyAllWindows()

print(objs)

