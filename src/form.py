import time
import os
import math
from datetime import datetime


class form:
    def __init__(self):
        self.current_time = time.strftime("%H-%M-%S", time.localtime())
        self.important_events = []
        self.important_num = 1
        root_dir = os.path.dirname(os.path.abspath(__file__))  # get the directory of the current file
        data_dir = os.path.join(root_dir, '../data')  # append 'data' to this directory
        self.path = os.path.join(data_dir, "data" + self.current_time + ".txt")  # append the filename to this directory
        f = open(self.path, 'w+')
        f.close()
        
    def add_important(self, txt):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        str = current_time + ": " + txt + "."
        self.important_events.append(str)
        self.important_num = self.important_num +1
             
        

    def print2file(self, events,firebase):
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
            str_to_firebase = str.replace("$", "")

        firebase.add_live(str_to_firebase)
        print(str_to_firebase + ": added")
        f.close()

    def print_report(self,firebase):
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
                        FMT = '%H:%M:%S'
                        if int(str((datetime.strptime(timestamp, FMT) - datetime.strptime(latest_event[2], FMT)))
                                .split(':')[1]) < 1:
                            latest_event[3] = latest_event[3] + 1
                            latest_event[2] = timestamp
                        else:
                            dict[event].append([latest_event[0] + 1, timestamp, timestamp, 1])
                    else:
                        dict[event] = [[1, timestamp, timestamp, 1]]  # [id, first timestamp, last timestamp, count]
                line = file.readline().rstrip()
                # print(split3)
        important_events_txt = "need to be updated later"
        in_total_txt=""
        situations_txt=""


        current_time = time.strftime("%H-%M-%S", time.localtime())
        # text = "report time: " + current_time + "\n\nImportant events:\n"
        report_time_txt=current_time
        text = report_time_txt + "\n\nImportant events:\n"
        # ADD HERE SUMMARY OF IMPORTANT EVENTS (DANGEROUS, SURPRISING...)
        text = text + "\n\nSituations:\n"
        # DOESN'T PRINT BY CHRONOLOGICAL ORDER!

        total_times = {}
        situations = []


        for t in dict:
            for con in dict[t]:                
                situations.append((str(t[:-1]), str(con[1]), str(con[2])))
                # text = text + str(t[:-1]) + " from " + str(con[1]) + " to " + str(con[2]) + ".\n"
                start = con[1]
                end = con[2]
                FMT = '%H:%M:%S'
                tdelta = datetime.strptime(end, FMT) - datetime.strptime(start, FMT)
                if t[:-1] in total_times:
                    t1 = datetime.strptime(str(total_times[t[:-1]][0]), FMT)
                    t2 = datetime.strptime(str(tdelta), FMT)
                    time_zero = datetime.strptime('00:00:00', FMT)
                    total_times[t[:-1]] = ((t1 - time_zero + t2).time(), total_times[t[:-1]][1] +1)
                else:
                    total_times[t[:-1]] = (tdelta, 1)

        # Sort situations and add to form:
        sorted_list = sorted(situations, key=lambda tim: datetime.strptime(tim[1], '%H:%M:%S'))
        for sit in sorted_list:
            added_str = str(sit[0]) + " from " + str(sit[1]) + " to " + str(sit[2]) + ".\n"
            situations_txt = situations_txt + added_str
            firebase.add_situations(added_str)

        text = text+situations_txt

        text = text + "\n\nIn total:\n"
        for t in total_times:
            firebase_str = str(t) + " for " + str(total_times[t][0])
            in_total_txt = in_total_txt + str(t) + " for " + str(total_times[t][0])
            split = str(total_times[t][0]).split(':')
            if split[0] != '0' and split[0] != '00':
                in_total_txt = in_total_txt + " hour(s), "
                firebase_str = firebase_str + " hour(s), "
            elif split[1] != '00':
                in_total_txt = in_total_txt + " minute(s), "
                firebase_str = firebase_str + " minute(s), "
            else:
                in_total_txt = in_total_txt + " second(s), "
                firebase_str = firebase_str + " second(s), "
                firebase.add_in_total(firebase_str)
            
            in_total_txt = in_total_txt + str(total_times[t][1]) + " time(s) total.\n"

        text = text + in_total_txt
        path = "../forms/form" + current_time + ".txt"
        file = open(path, 'w+')
        file.write(text)
        file.close()

        for event in self.important_events:
            firebase.add_important(event)
            
        firebase.update_form(report_time_txt,in_total_txt,situations_txt,important_events_txt)



        # os.startfile("C:/Users/tamir/OneDrive/summerProjects/situations_detector/"+path)









