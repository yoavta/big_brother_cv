import time

import pyrebase
from firebase_config import firebase_config
firebase = firebase_config()


db = firebase.firebase.database()
data = {"name": "Parwiz Forogh"}

# db.child("power").set("19")
while (True):
    mode = db.child("power").get().val()
    if (mode == '0'):
        print ('camera off')
    else:
        print("camera on")
    time.sleep(2)
