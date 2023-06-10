import time

import cv2

from categories_list import CategoriesList
from configuration import Configuration
from controller_client import ActionsTypes
from controller_client import send_data
from controller_server import run_server, accept_connection, process_data
from firebase_config import FirebaseConfig
from form import form
from src.Bbox import BBox
from src.DetectorStore import DetectorStore
from src.data_proceesor import analyze_connections
from utils import speak, camera_move_check

configuration = Configuration()


def initial():
    run_server()
    send_data(ActionsTypes.TURN_ON)
    accept_connection()


def main():
    # creating objects
    firebase = FirebaseConfig()

    # firebase configurations
    db = firebase.firebase
    categories = CategoriesList(firebase, db)
    configuration.add_categories(categories)
    first_person_show = False

    data_form = form()
    detectors_store = DetectorStore()

    while True:
        print("waiting for power on to get started")
        while not firebase.is_on():
            time.sleep(2)
            pass

        initial()
        user_name = firebase.get_name()

        print("capture started")
        firebase.reset_data()
        move_direction = None

        while True:
            class_ids = []
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
                send_data(ActionsTypes.TURN_OFF)
                break

            is_person = False

            for i in range(configuration.number_of_iterations):
                bboxes = []
                frame = process_data()
                if frame is None:
                    continue

                cv2_im = frame.copy()

                threshold = 0.3
                objs = detectors_store.object_detector.get_objects(frame, threshold, detectors_store.object_detector,
                                                                   cv2_im)

                hands, cv2_im = detectors_store.hand_detector.get_model().findHands(cv2_im)
                cv2_im, person_center, person_box = detectors_store.append_objs_to_img(cv2_im, objs, class_ids, bboxes,
                                                                                       hands,
                                                                                       possible_connections)
                is_person = person_center is not None or is_person
                if first_person_show == False and person_center is not None:
                    first_person_show = True
                    st = user_name + " has entered the house."
                    events = [st]
                    data_form.print2file(events, firebase)
                    speak("hello" + user_name)

                if person_center is not None:
                    current_move = camera_move_check(cv2_im, configuration.moving_sensitivity, person_center,
                                                     person_box, configuration.screen_width,
                                                     configuration.screen_height)
                    if current_move:
                        send_data(current_move)
                    move_direction = current_move
                else:
                    move_direction = None

                cv2.imshow("Image", cv2_im)
            for key in possible_connections.keys():
                connections.append(key)

            analyze_connections(connections, data_form, user_name, firebase, configuration.categories)
            if is_person and move_direction is None:
                send_data(ActionsTypes.STOP)
            elif move_direction is None and not is_person:
                send_data(ActionsTypes.START)

        print("capture stopped")


if __name__ == '__main__':
    main()
