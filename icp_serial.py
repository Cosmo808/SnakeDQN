import time
import serial
import math


class icpReader:
    def __init__(self):
        self.ser = serial.Serial()
        self.speed = {'X': 0, 'Y': 0, 'Z': 0}
        self._init()

    def _init(self):
        self.ser.baudrate = 115200
        self.ser.port = '/dev/ttyUSB0'
        self.ser.open()

    def read_speed(self):
        self.ser.write('r'.encode('utf-8'))
        info = self.ser.read_all()
        info = str(info)[2:-1].split(' ')
        for speed_dir in info:
            try:
                speed = int(speed_dir[2:])
            except ValueError:
                speed = 0
            if 'X' in str(speed_dir):
                self.speed['X'] = speed
            elif 'Y' in str(speed_dir):
                self.speed['Y'] = speed
            elif 'Z' in str(speed_dir):
                self.speed['Z'] = speed
        time.sleep(0.5)
        return self.speed

    @staticmethod
    def speed2vector(speed):
        x = speed['X']
        y = speed['Y']
        z = speed['Z']

        radius = math.sqrt(x * x + y * y)
        if not radius:
            return [0, 0]
        cos = x / radius
        theta = math.acos(cos) / math.pi * 180
        if y < 0:
            theta += 180
        theta -= 45
        if theta < 0:
            theta += 360
        delta = z / radius
        theta += delta

        vector = [radius, theta]
        return vector

    @staticmethod
    def vector2action(vector):
        radius = vector[0]
        theta = vector[1]
        if theta > 180:
            theta -= 360
        if -60 <= theta < 60:
            action = [1, 0, 0]
        elif 60 <= theta < 180:
            action = [0, 1, 0]
        else:
            action = [0, 0, 1]
        # [straight, right, left]
        return action

    def __del__(self):
        self.ser.close()
