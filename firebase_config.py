import pyrebase


class firebase_config:

    def __init__(self):
        self.firebase = self.config()


    def config(self):
        config = {
            "apiKey": "AIzaSyCQ03X8JsqHjcj7cjxn7aadh7LInoBWK50",
            "authDomain": "big-brother-cv.firebaseapp.com",
            "projectId": 'big-brother-cv',
            "databaseURL": "https://big-brother-cv-default-rtdb.firebaseio.com/",
            "messagingSenderId": "880755436451",
            "appId": "1:880755436451:web:1ebd63181e4eedd7890816",
            "measurementId": "G-WPGXX4BPBK",
            "storageBucket": "big-brother-cv.appspot.com"
        }

        return pyrebase.initialize_app(config)

    def is_changed(self):
        if self.firebase.database().child("data").child("categories").child("change").get().val()==1:
            return True
        return False

    def update_finished(self):
        self.firebase.database().child("data").child("categories").child("change").set(0)

    def update(self, dic):
        list = []
        for i in dic:
            if dic.get(i)== 1:
                list.append(i)

        return list

