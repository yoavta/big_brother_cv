from utils import read_label_file, read_file, get_path

class Configuration:
    def __init__(self):
        self.classNames = read_file(get_path("../Resources/coco.names.txt"))
        self.person_list = read_file(get_path("../Resources/person.txt"))
        self.hand_list = read_file(get_path("../Resources/hand.txt"))
        self.other_list = read_file(get_path("../Resources/other.txt"))
        self.default_labels = 'coco_labels.txt'
        self.labels = read_label_file(get_path('../Resources/coco.names.txt'))
        self.screen_width = 640
        self.screen_height = 480
        self.number_of_iterations = 1

    def add_categories(self, categories):
        self.categories = categories
