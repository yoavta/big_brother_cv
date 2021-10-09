import time

import yeelight
from yeelight import *
from yeelight import transitions

try:
    bulb1 = Bulb('XXX.XXX.XXX.XXX', effect="smooth")
    bulb2 = Bulb('XXX.XXX.XXX.XXX', effect="smooth")
    bulbs = [bulb1, bulb2]
except:
    print ("error with connection to cam")
old_prop = []

def alarm_once(sec):
    try:
        alarm_all()
        time.sleep(sec)
        stop_alarm_all()
    except:
        print ("error with connection to cam")


def alarm_all():
    flow = Flow(

        action=Flow.actions.recover,
        transitions=transitions.alarm(duration=50)
    )

    for b in bulbs:
        old_prop.append(b.get_properties().get('power')=='on')
        if b.last_properties.get('power') == 'off':
            b.turn_on()
        b.start_flow(flow)


def stop_alarm_all():
    flow = Flow(

        action=Flow.actions.recover,
        transitions=transitions.alarm(duration=50)
    )

    for b in bulbs:
        b.stop_flow(flow)

    set_properties()



def set_properties():
    properties = ['power', 'bright', 'ct', 'rgb', 'hue', 'sat', 'color_mode', 'flowing', 'delayoff', 'music_on', 'name',
                  'bg_power', 'bg_flowing', 'bg_ct', 'bg_bright', 'bg_hue', 'bg_sat', 'bg_rgb', 'nl_br', 'active_mode']


    index = 0
    for b in bulbs:
        if old_prop[index] == 0:
            b.turn_off()
        else:
            b.turn_off()
            b.turn_on()
        index = index + 1


