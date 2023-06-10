# linking object connection to a categories and produce update text.
from src.utils import speak


def analyze_connections(connections, data_form, user_name, firebase,categories):
    events = []
    for con in connections:
        list_in = categories.which_list_am_i_complete(con)
        if list_in == "computer":
            st = user_name + " is using the computer with a(n) " + con + "."
        elif list_in == "danger":
            st = user_name + " is playing with a " + con + "."
            if con in categories.get_importants():
                st = "watch out!! " + user_name + " is playing with a(n) " + con + "."
                # speaking and open warning lights.
                speak(st)
                # Lights.alarm_once(3)
        elif list_in == "food":
            st = user_name + " is eating a(n) " + con + "."
        elif list_in == "holdings":
            st = user_name + " is holding a(n) " + con + "."
        elif list_in == "playing":
            st = user_name + " is playing with a(n) " + con + "."
        elif list_in == "sitting":
            st = user_name + " is sitting on a(n) " + con + "."
        elif list_in == "specific":
            if con == 'toothbrush':
                st = user_name + "is brushing her teeth."
            elif con == 'book':
                st = user_name + " is reading a book."
            elif con == 'cell phone':
                st = user_name + " is using her phone."
        elif list_in == "tv":
            st = user_name + " is using the tv with a(n)" + con + "."
        elif list_in == "wearing":
            st = user_name + " is wearing a(n) " + con + "."

        else:
            st = user_name + " has a connection with a(n) " + con + "."
        events.append(st)

        # check if this connection eas nark as important.
        if con in categories.get_importants():
            data_form.add_important(st)

    # printing to file
    data_form.print2file(events, firebase)

    return events
