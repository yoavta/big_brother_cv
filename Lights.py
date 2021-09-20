import time

import yeelight
from yeelight import *
from yeelight import transitions

bulb1 = Bulb('192.168.0.191', effect="smooth")
bulb2 = Bulb('192.168.0.159', effect="smooth")
# bulb_savta = Bulb('192.168.1.167', effect="smooth")
bulbs = [bulb1, bulb2]

old_prop = []

def alarm_once(sec):
    alarm_all()
    time.sleep(sec)
    stop_alarm_all()

def print_hi(name):
    bulb_savta.turn_off()
#     from yeelight import discover_bulbs
#     bulbs =  discover_bulbs()
#     print(bulbs)
    #
    # # transitions = [
    # #     TemperatureTransition(20, duration=500),
    # #     SleepTransition(duration=50),
    # #     TemperatureTransition(10000, duration=500)
    # # ]
    #
    # flow = Flow(
    #     count=10,
    #     action=Flow.actions.recover,
    #     transitions=transitions.alarm(duration=50)
    # )
    #
    # bulb1 = Bulb('192.168.0.191')
    # bulb2 = Bulb('192.168.0.159')
    # bulb1.turn_on()
    # bulb1.start_flow(flow)
    # bulb1.start_flow(flow)
    # time.sleep(10)
    # bulb1.turn_off()
# 
#     alarm_all()
#     time.sleep(5)
#     stop_alarm_all()



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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
