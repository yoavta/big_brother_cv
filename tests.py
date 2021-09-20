import time

import pyrebase
from firebase_config import firebase_config
firebase = firebase_config()


db = firebase.db


db.child("power").set("1")
while (True):
    mode = db.child("power").get().pyres
    if (mode == '0'):
        print ('camera off')
    else:
        print("camera on")
    time.sleep(2)
