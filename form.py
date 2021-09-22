import time
import datetime
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
                        time_list = timestamp.split(':')
                        latest_event = dict[event][-1]
                        if (int(time_list[0]) - int(latest_event[2].split(':')[0]) == 0 and
                            int(time_list[1]) - int(latest_event[2].split(':')[1]) <= 5) or\
                            (int(time_list[0]) - int(latest_event[2].split(':')[0]) == 1 and
                                abs(int(time_list[1]) - int(latest_event[2].split(':')[1])) >= 55) or\
                                abs((int(time_list[0]) - int(latest_event[2].split(':')[0])) == 23 and
                                    abs(int(time_list[1]) - int(latest_event[2].split(':')[1])) <= 5):
                            latest_event[3] = latest_event[3] + 1
                            latest_event[2] = timestamp
                        else:
                            dict[event].append([latest_event[0] + 1, timestamp, timestamp, 1])
                    else:
                        dict[event] = [[1, timestamp, timestamp, 1]]  # [id, first timestamp, last timestamp, count]
                line = file.readline().rstrip()
                # print(split3)

        current_time = time.strftime("%H-%M-%S", time.localtime())
        text = "report time: " + current_time + "\n\nImportant events:\n"
        # ADD HERE SUMMARY OF IMPORTANT EVENTS (DANGEROUS, SURPRISING...)
        text = text + "\n\nSituations:\n"

        total_times = {}

        for t in dict:
            for con in dict[t]:
                text = text + str(t[:-1]) + " from " + str(con[1]) + " to " + str(con[2])+".\n"
                # start = con[1].split(':')
                # end = con[2].split(':')
                # hours =  int(end[0]) - int(start[0])
                # minutes = int(end[1]) - int(start[1])
                # if hours == 0:
                #     total_times[t] = minutes
                # elif hours == 1:
                #     total_times[t] = abs(minutes)
                # if (int(end[0]) - int(start[0]) == 0 and
                #     int(end[1]) - int(start[1]) <= 5) or \
                #         (int(time_list[0]) - int(latest_event[2].split(':')[0]) == 1 and
                #          abs(int(time_list[1]) - int(latest_event[2].split(':')[1])) >= 55) or \
                #         abs((int(time_list[0]) - int(latest_event[2].split(':')[0])) == 23 and
                #             abs(int(time_list[1]) - int(latest_event[2].split(':')[1])) <= 5):

        text = text + "\n\nIn total:\n"
        for t in dict:
            text = text + str(t[:-1]) + " for " + " ______ hour(s)."

        path = "forms/form" + current_time + ".txt"
        file = open(path, 'w+')
        file.write(text)
        file.close()
        # os.startfile("C:/Users/tamir/OneDrive/summerProjects/situations_detector/"+path)










