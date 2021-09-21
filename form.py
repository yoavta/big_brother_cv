import time
import os
import math


class form:
    def __init__(self):
        self.current_time = time.strftime("%H-%M-%S", time.localtime())

        self.path = "data/data" + self.current_time + ".txt"
        f = open(self.path, 'w+')
        f.close()

    def print2file(self, events):
        if not events:
            return

        # data_file = "data.txt"
        # f = open(data_file, 'w')
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        str = current_time + ";"
        for event in events:
            str = str + " " + event + "$"

        with open(self.path, "a") as f:
            f.write(str + "\n")
        print(str)
        f.close()

    def print_report(self):
        # f = open(self.path, 'r')
        dict = {}
        with open(self.path, 'r', encoding='UTF-8') as file:
            line = file.readline().rstrip()
            while line:
                split1 = line.split(';')
                split2 = split1[1]
                split3 = split2.split('$')
                split4 = split3[:-1]
                timestamp = split1[0]
                for event in split4:
                    if event in dict.keys():
                        # Check timestamp to see if it is a new connection:
                        if abs(int(timestamp.split(':')[1]) - int(dict[event][-1][1].split(':')[1])) <= 5:
                            dict[event][-1][2] = dict[event][-1][2] + 1
                            dict[event][-1][1] = timestamp
                        else:
                            dict[event].append([dict[event][-1][0] + 1, timestamp, 1])
                    else:
                        dict[event] = [[1, timestamp, 1]]  # [id, timestamp, count]
                line = file.readline().rstrip()
                # print(split3)

        current_time = time.strftime("%H-%M-%S", time.localtime())
        text = "report time: "+current_time+".\n\nSituations:\n"

        for t in dict:
            for con in dict[t]:
                text = text + str(t[:-1]) + "(" + str(con[0]) + "):  "+str(con[2])+" times.\n"

        path = "forms/form" + current_time + ".txt"
        file = open(path, 'w+')
        file.write(text)
        file.close()
        # os.startfile("C:/Users/tamir/OneDrive/summerProjects/situations_detector/"+path)










