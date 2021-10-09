
class categories_list:
    
    def __init__(self, firebase,db):
        self.db = db
        self.firebase = firebase
        self.sitting_list = firebase.update_always(db.child("data").child("categories").child("sitting").get().val())
        self.computer_list = firebase.update_always(db.child("data").child("categories").child("computer").get().val())
        self.holdings_list = firebase.update_always(db.child("data").child("categories").child("holdings").get().val())
        self.playing_list = firebase.update_always(db.child("data").child("categories").child("playing").get().val())
        self.specific_list = firebase.update_always(db.child("data").child("categories").child("specific").get().val())
        self.tv_list = firebase.update_always(db.child("data").child("categories").child("tv").get().val())
        self.using_list = firebase.update_always(db.child("data").child("categories").child("using").get().val())
        self.danger_list = firebase.update_always(db.child("data").child("categories").child("danger").get().val())
        self.food_list = firebase.update_always(db.child("data").child("categories").child("food").get().val())
        self.wearing_list = firebase.update_always(db.child("data").child("categories").child("wearing").get().val())
        self.complete_lists = [self.sitting_list,self.computer_list,self.holdings_list,self.playing_list,self.specific_list,self.tv_list,self.using_list,self.danger_list,self.food_list,self.wearing_list]
        self.complete_list= self.make_one_list(self.complete_lists)


        self.sitting_list_important = firebase.update(db.child("data").child("categories").child("sitting").get().val())
        self.computer_list_important = firebase.update(db.child("data").child("categories").child("computer").get().val())
        self.holdings_list_important = firebase.update(db.child("data").child("categories").child("holdings").get().val())
        self.playing_list_important = firebase.update(db.child("data").child("categories").child("playing").get().val())
        self.specific_list_important = firebase.update(db.child("data").child("categories").child("specific").get().val())
        self.tv_list_important = firebase.update(db.child("data").child("categories").child("tv").get().val())
        self.using_list_important = firebase.update(db.child("data").child("categories").child("using").get().val())
        self.danger_list_important = firebase.update(db.child("data").child("categories").child("danger").get().val())
        self.food_list_important = firebase.update(db.child("data").child("categories").child("food").get().val())
        self.wearing_list_important = firebase.update(db.child("data").child("categories").child("wearing").get().val())
        self.updated_categories_lists =  [self.sitting_list_important,self.computer_list_important,self.holdings_list_important,self.playing_list_important,self.specific_list_important,self.tv_list_important,self.using_list_important,self.danger_list_important,self.food_list_important,self.wearing_list_important]
        self.updated_categories_list= self.make_one_list(self.updated_categories_lists)


    def update_lists(self):
        db = self.db
        firebase = self.firebase
        self.sitting_list_important = firebase.update(db.child("data").child("categories").child("sitting").get().val())
        self.computer_list_important = firebase.update(db.child("data").child("categories").child("computer").get().val())
        self.holdings_list_important = firebase.update(db.child("data").child("categories").child("holdings").get().val())
        self.playing_list_important = firebase.update(db.child("data").child("categories").child("playing").get().val())
        self.specific_list_important = firebase.update(db.child("data").child("categories").child("specific").get().val())
        self.tv_list_important = firebase.update(db.child("data").child("categories").child("tv").get().val())
        self.using_list_important = firebase.update(db.child("data").child("categories").child("using").get().val())
        self.danger_list_important = firebase.update(db.child("data").child("categories").child("danger").get().val())
        self.food_list_important = firebase.update(db.child("data").child("categories").child("food").get().val())
        self.wearing_list_important = firebase.update(db.child("data").child("categories").child("wearing").get().val())
        self.updated_categories_lists =  [self.sitting_list_important,self.computer_list_important,self.holdings_list_important,self.playing_list_important,self.specific_list_important,self.tv_list_important,self.using_list_important,self.danger_list_important,self.food_list_important,self.wearing_list_important]
        self.updated_categories_list= self.make_one_list(self.updated_categories_lists)
        
    def which_list_am_i_complet(self, item):
        if item in self.sitting_list:
            return "sitting"
        
        if item in self.computer_list :
            return "computer"
        
        elif item in self.holdings_list:
            return "hodlings"
        
        elif item in self.playing_list:
            return "playing"
        
        elif item in self.specific_list:
            return "specific"
        
        elif item in self.tv_list:
            return "tv"
        
        elif item in self.using_list:
            return "using"
        
        elif item in self.danger_list:
            return "danger"
        
        elif item in self.food_list:
            return "food"
        
        elif item in self.wearing_list:
            return "wearing"
        
    def which_list_am_i_updated(self, item):
        if item in self.sitting_list_important:
            return "sitting"
        
        if item in self.computer_list_important :
            return "computer"
        
        elif item in self.holdings_list_important:
            return "hodlings"
        
        elif item in self.playing_list_important:
            return "playing"
        
        elif item in self.specific_list_important:
            return "specific"
        
        elif item in self.tv_list_important:
            return "tv"
        
        elif item in self.using_list_important:
            return "using"
        
        elif item in self.danger_list_important:
            return "danger"
        
        elif item in self.food_list_important:
            return "food"
        
        elif item in self.wearing_list_important:
            return "wearing"
        
    def make_one_list(self,lists):
        new_list = []
        for lst in lists:
            for item in lst:
                new_list.append(item)
        return new_list
    
    def get_importants(self):
        return self.updated_categories_list
    

            
        
        
        
        
        