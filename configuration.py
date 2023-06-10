from utils import read_label_file


class Configuration:
    def __init__(self):
        self.classNames = self.read_file("Resources/coco.names.txt")
        self.person_list = self.read_file("Resources/person.txt")
        self.hand_list = self.read_file("Resources/hand.txt")
        self.other_list = self.read_file("Resources/other.txt")
        self.default_labels = 'coco_labels.txt'
        self.labels = read_label_file('Resources/coco.names.txt')
        self.screen_width = 640
        self.screen_height = 480
        self.NUMBER_OF_ITERATIONS = 1

    def read_file(self, file_path):
        with open(file_path, 'rt') as f:
            return f.read().rstrip('\n').split('\n')
