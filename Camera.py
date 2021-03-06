import pantilthat
import time
class Camera:
    def __init__(self,start_x, start_y):
        self.current_x = start_x
        self.current_y = start_y
        self.max_left=-87
        self.max_right=87
        self.max_up=87
        self.max_down=-87
        self.servo= pantilthat.PanTilt()
        self.servo.pan(self.current_x)
        self.servo.tilt(self.current_y)
#         self.show_off()
    
        
        
    def show_off(self):
        self.move_right(2)
        time.sleep(0.5)
        self.move_right(2)
        time.sleep(0.5)
        self.move_down(2)
        self.move_right(2)
        time.sleep(0.5)
        self.move_down(2)
        self.move_right(2)
        time.sleep(0.2)
        self.move_down(2)
        self.move_right(2)
        time.sleep(0.2)
        self.move_down(2)
        self.move_right(2)
        self.move_right(2)
        self.move_right(2)
        self.move_down(2)
        self.move_right(3)
        
    def camera_move(self,dirc, dis):
        str_dis = str(dis)[:-2]
        size = 2
        if dirc == 1:
            print("camera moving "+ str_dis+ " left")
            self.move_left(size)
        elif dirc == 2:
            print("camera moving "+ str_dis+ " right")
            self.move_right(size)
        elif dirc == 3:
            print("camera moving "+ str_dis+ " up")
            self.move_up(size)
        else:
            print("camera moving "+ str_dis+ " down")
            self.move_down(size)


    def move_right(self,dis):
        if self.current_x-dis>=self.max_right-dis:
            return False
        self.current_x= self.current_x-dis
        self.servo.pan(self.current_x)
        return True
    
    def move_left(self,dis):
        if self.current_x+dis<=self.max_left-dis:
            return False
        self.current_x= self.current_x+dis
        self.servo.pan(self.current_x)
        return True
    
    def move_up(self,dis):
        if self.current_y-dis>=self.max_up-dis:
            return False
        self.current_y= self.current_y-dis
        self.servo.tilt(self.current_y)
        return True
    
    def move_down(self,dis):
        if self.current_y+dis<=self.max_down+dis:
            return False
        self.current_y= self.current_y+dis
        self.servo.tilt(self.current_y)
        return True
        
