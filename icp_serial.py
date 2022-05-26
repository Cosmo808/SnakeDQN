import serial
import time


class icpReader:
    def __init__(self):
        self.ser = serial.Serial()
        self.speed = {}
        self._init()

    def _init(self):
        self.speed = {}
        self.ser.baudrate = 115200
        self.ser.port = '/dev/ttyUSB0'
        self.ser.open()

    def read_speed(self):
        self.ser.write('r'.encode('utf-8'))
        # for i in range(3):
        #     s = self.ser.readline()  # X:123
        #     try:
        #         speed = int(str(s)[4:-3])
        #     except ValueError:
        #         speed = 0
        #     if 'X' in str(s):
        #         self.speed['X'] = speed
        #     elif 'Y' in str(s):
        #         self.speed['Y'] = speed
        #     elif 'Z' in str(s):
        #         self.speed['Z'] = speed
        return self.ser.readline()

    def __del__(self):
        self.ser.close()


if __name__ == '__main__':
    icp_reader = icpReader()
    while True:
        ss = icp_reader.read_speed()
        print(ss)
        time.sleep(1)
