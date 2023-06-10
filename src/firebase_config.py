import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


class FirebaseConfig:

    def __init__(self):
        self.firebase = self.config()
        self.event_num = 1
        self.in_total_num = 1
        self.situations_num = 1
        self.important_num = 1
        self.reset_data()

    def config(self):
        cred = credentials.Certificate('../big-brother-cv-firebase-adminsdk-ub4bk-de7418174c.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://big-brother-cv-default-rtdb.firebaseio.com/'
        })
        return db

    def is_changed(self):
        if db.reference("data/categories/change").get() == 1:
            return True
        return False

    def tick_categories(self, name, mark):
        categories = db.reference(f"data/categories/{name}").get()
        for key in categories:
            db.reference(f"data/categories/{name}/{key}").set(mark)

    def update_finished(self):
        db.reference("data/categories/change").set(0)

    def update(self, dic):
        return [key for key, value in dic.items() if value == 1]

    def update_always(self, dic):
        return list(dic.keys())

    def update_form(self, report_time_txt, in_total_txt, situations_txt, important_events_txt):
        db.reference("data/forms").update({"report time": report_time_txt})

        db.reference("produce report").set(0)

    def add_live(self, txt):
        db.reference(f"data/live update/{self.event_num}").set(txt)
        self.event_num += 1

    def add_important(self, txt):
        db.reference(f"data/forms/important events/{self.important_num}").set(txt)
        self.important_num += 1

    def add_in_total(self, txt):
        db.reference(f"data/forms/in total/{self.in_total_num}").set(txt)
        self.in_total_num += 1

    def add_situations(self, txt):
        db.reference(f"data/forms/situations/{self.situations_num}").set(txt)
        self.situations_num += 1

    def reset_data(self):
        db.reference("data/live update").set({"pass": "pass"})
        db.reference("data/forms/in total").set({"pass": "pass"})
        db.reference("data/forms/situations").set({"pass": "pass"})
        db.reference("data/forms/important events").set({"pass": "pass"})

    def is_on(self):
        return db.reference("power").get() == 1

    def is_report(self):
        return db.reference("produce report").get() == 1

    def upload_img(self, file_name):
        storage = firebase_admin.storage.bucket()
        blob = storage.blob(f"images/{file_name}")
        blob.upload_from_filename(file_name)
        os.remove(file_name)

    def get_name(self):
        return db.reference("name").get()
