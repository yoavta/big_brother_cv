from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import pantilthat
from time import sleep

class Camera:
    def __init__(self,start_tilt,start_pan):
        self.remote_factory = PiGPIOFactory(host='xxx.xxx.xx.x.xxx')
        self.pan = Servo(14,pin_factory=self.remote_factory)
        self.tilt = Servo(18,pin_factory=self.remote_factory)
        self.move_pan_to(start_pan)
        self.move_tilt_to(start_tilt)
        self.step = 0.05


    def move_pan_to(self,to):
        if to<=1 and to>=-1:
            self.pan.value=to
            return True
        else:
            return False

    def move_tilt_to(self,to):
        if to<=1 and to>=-1:
            self.tilt.value=to
            return True
        else:
            return False

    def move_pan_one_step(self,direction):
        return self.move_pan_to(self.pan.value+ direction*self.step)

    def move_tilt_one_step(self,direction):
        return self.move_tilt_to(self.tilt.value+ direction*self.step)


