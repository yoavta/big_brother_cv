import pyrebase
import time
import os


class firebase_config:

    def __init__(self):
        self.firebase = self.config()
        self.event_num = 1
        self.in_total_num =1
        self.situations_num = 1
        self.important_num = 1
        self.initial()
        
    



    def config(self):
        config = {
            "apiKey": "AIzaSyCQ03X8JsqHjcj7cjxn7aadh7LInoBWK50",
            "authDomain": "big-brother-cv.firebaseapp.com",
            "projectId": 'big-brother-cv',
            "databaseURL": "https://big-brother-cv-default-rtdb.firebaseio.com/",
            "messagingSenderId": "880755436451",
            "appId": "1:880755436451:web:1ebd63181e4eedd7890816",
            "measurementId": "G-WPGXX4BPBK",
            "storageBucket": "big-brother-cv.appspot.com",
            "serviceAccount": "/home/pi/big_brother_cv/Resources/big-brother-cv-firebase-adminsdk-ub4bk-68309ff742.json"

        }

        return pyrebase.initialize_app(config)

    def is_changed(self):
        if self.firebase.database().child("data").child("categories").child("change").get().val()==1:
            return True
        return False

    def tick_categories(self, name, mark):
        list = self.firebase.database().child("data").child("categories").child(name).get().val().keys()
        for l in list:
            self.firebase.database().child("data").child("categories").child(name).update({l:mark})
            


    def update_finished(self):
        self.firebase.database().child("data").child("categories").child("change").set(0)

    def update(self, dic):
        list = []
        for i in dic:
            if dic.get(i)== 1:
                list.append(i)

        return list
    
    def update_always(self, dic):
        list = []
        for i in dic:
            list.append(i)

        return list

    def update_form(self,report_time_txt,in_total_txt,situations_txt,important_events_txt):
        self.firebase.database().child("data").child("forms").update({"report time":report_time_txt})
#         self.firebase.database().child("data").child("forms").update({"important events":important_events_txt})
#         time.sleep(1)
        self.firebase.database().child("produce report").set(0)

#         self.firebase.database().child("data").child("forms").update({"situations":situations_txt})
#         self.firebase.database().child("data").child("forms").update({"in total":in_total_txt})

    def add_live(self, txt):
        self.firebase.database().child("data").child("live update").update({self.event_num:txt})
        self.event_num=self.event_num+1
        
    def add_important(self, txt):
        self.firebase.database().child("data").child("forms").child("important events").update({self.important_num:txt})
        self.important_num=self.important_num+1

        
    def add_in_total(self, txt):
        self.firebase.database().child("data").child("forms").child("in total").update({self.in_total_num:txt})
        self.in_total_num = self.in_total_num +1
        
    def add_situations(self, txt):
        self.firebase.database().child("data").child("forms").child("situations").update({self.situations_num:txt})
        self.situations_num = self.situations_num +1

        
    def initial(self):
        self.firebase.database().child("data").child("live update").set({"pass":"pass"})
        self.firebase.database().child("data").child("forms").child("in total").set({"pass":"pass"})
        self.firebase.database().child("data").child("forms").child("situations").set({"pass":"pass"})
        self.firebase.database().child("data").child("forms").child("important events").set({"pass":"pass"})
        
        for i in range (self.firebase.storage().child("images").list_files().__sizeof__()):
            try:
                path = "images/important"+str(i)+".jpg"
                self.firebase.storage().child("images").delete(path)
            except:
                pass
        

        
    def is_on(self):
        if self.firebase.database().child("power").get().val()==1:
            return True
        return False
    
    def is_report(self):
        if self.firebase.database().child("produce report").get().val()==1:
            return True
        else:
            return False
        
    def upload_img(self,file_name):
        storage = self.firebase.storage()
        storage.child("images").child(file_name).put(file_name)
        os.remove(file_name)
        
        
        
        

        

      
