class CategoriesList:

    def __init__(self, firebase, db):
        self.db = db
        self.firebase = firebase
        self.categories_names = ["sitting", "computer", "holdings", "playing", "specific", "tv", "using", "danger",
                                 "food", "wearing"]
        self.complete_lists = self.get_complete_lists()
        self.complete_list = self.make_one_list(self.complete_lists)
        self.updated_categories_lists = self.get_updated_categories_lists()
        self.updated_categories_list = self.make_one_list(self.updated_categories_lists)

    def get_complete_lists(self):
        return [self.firebase.update_always(self.db.reference(f"data/categories/{category}").get()) for category in
                self.categories_names]

    def get_updated_categories_lists(self):
        return [self.firebase.update(self.db.reference(f"data/categories/{category}").get()) for category in
                self.categories_names]

    def update_lists(self):
        self.updated_categories_lists = self.get_updated_categories_lists()
        self.updated_categories_list = self.make_one_list(self.updated_categories_lists)

    def which_list_am_i(self, item, lists):
        for category, lst in zip(self.categories_names, lists):
            if item in lst:
                return category
        return None

    def make_one_list(self, lists):
        new_list = []
        for lst in lists:
            for item in lst:
                new_list.append(item)
        return new_list

    def get_importants(self):
        return self.updated_categories_list
