# _*_ coding: utf-8 _*_

# 3/30

# from MSP430 import MSPComm
# from MSP430 import WINDOW_SIZE
import socket
import time
import threading
import serial.tools.list_ports
import serial
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.rcsetup as rcsetup
import math
import os
import copy
import cv2
# from sklearn.neighbors import NearestNeighbors
# import feature_extraction8
# from sklearn import preprocessing
# from sklearn.externals import joblib
# from painter import *
# For W & R CSV file
import csv
import random

# matplotlib.use('TkAgg')
# print(rcsetup.all_backend s)
INTERNAL_MUX_MODE = 0
EXTERNAL_MUX_MODE = 1
# WINDOW_SIZE = 10
WINDOW_SIZE = 10
MSP_CHANNEL = 4
CONDUCTIVE_THRESHOLD = 600000
NON_CONDUCTIVE_PEAK = 600000
CONDUCTIVE_PEAK = 5000000  # TO ++
CAP_DECREASE_PEAK = 100000

NON_NOISE_THRESHOLD = 10000
SINGLE_CAP_THRESHOLD = 5000000
ARDUINO_SERIAL_PORT = "COM4"

CHANELL = 4
SUPPORTED_ACTION_NUM = 100
ACTION_ENERGY_THRESHOLD = 10000
OBJECT_ENERGY_THRESHOLD = 1000
MAX_DRAW_POINT = 144    # self.c * self.r
AREA_PERCENT = 0.7
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800


def serialRead():
    # detect serial port.
    serialDict = []
    serialNum = 0
    ports = list(serial.tools.list_ports.comports())
    for a, b, c in ports:
        # print(a,b,c)
        d = a.split('.')
        if d[-1] != 'Bluetooth-Incoming-Port':
            # print (d[-1])
            serialDict.append(a)
            serialNum += 1
    print("Serial Read Success.")
    return serialNum, serialDict


class FetchData:
    def __init__(self):
        # second
        self.mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mysocket.bind(('localhost', 5000))
        self.mysocket.listen(1)
        self.conn = None
        self.r = 12
        self.c = 12
        self.recalibration = False
        self.calibration = True
        self.totalChannel = self.r * self.c
        self.data = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.base = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.data_p = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.base_p = np.zeros((self.totalChannel, WINDOW_SIZE))

        self.object_set = ["wet_flower", "cold_water"]
        self.trainNumber = 10
        self.trainCount = 0

        self.index = 0
        self.action = np.zeros((SUPPORTED_ACTION_NUM))
        self.energy_metrix = np.zeros((self.r, self.c))
        self.pos_metrix = np.zeros((self.r, self.c))

        self.position = np.zeros((SUPPORTED_ACTION_NUM))  # the current poition(X*self.r+Y) of object(index)
        self.unknownPositionEnergy = np.zeros((self.r * self.c))
        # self.move = np.zeros((SUPPORTED_ACTION_NUM))  # if the position stays unchanged, move = false, else true
        self.track = np.array(self.position, dtype=np.str)  # the position history of object(index)

        self.lastData = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.objectEnergy = np.zeros((SUPPORTED_ACTION_NUM, self.totalChannel))
        self.min_energy = 0
        self.last_down_data = 0
        self.point_set = []
        self.mv_energy = 0
        self.last_mv_pos = 0
        self.v_time = 0

        # time.sleep(1)
        self.start_object_recog = False
        self.diffs_p = np.zeros((self.totalChannel))

        # time.sleep(1)
        self.reset()

        is_setup = False
        while not is_setup:
            try:
                self.arduino_port = serial.Serial(port=ARDUINO_SERIAL_PORT, baudrate=115200)
                is_setup = True
            except Exception as e:
                print(e)
                print("Wait for device")
        is_setup = False
        while not is_setup:
            self.arduino_port.reset_input_buffer()
            self.arduino_port.dsrdtr = 0
            # self.arduino_port.write("reset\n".encode())
            string_ = self.arduino_port.readline()
            print(string_)
            is_setup = "SUCCESS" in str(string_)
            print("Wait for setup")
        print('build successful')

    def socket_handler(self):
        while True:
            if self.conn:
                data = self.conn.recv(1024)  # receive data from the socket(self)
                # return value is a string representing data received. Max of data to be received at once is 1024
                if data:    # succeed in receiving data
                    d = data.decode("utf-8")
                    if d.strip() == "reset":    # R-key on keyboard is pressed
                        # s_beforeC = data
                        # self.getCurrent()   # fresh cur[k][i], to save data before changed
                        self.reset()  # Set self.recalibration = True, self.diffs = [0]
                        # self.getChanged()   # get actual signal after placing objects
                        self.start_object_recog = True
                    elif d.strip() == "log":    # L-key on keyboard is pressed
                        print("             log")
                        self.fetch_ml_data(self.object_set[self.trainCount//self.trainNumber])
                    elif d.strip() == "peak_maker":  # P-key on keyboard is pressed
                        print("             save peak in file")
                        self.save_peak()
            else:
                self.conn, addr = self.mysocket.accept()
                if self.conn:
                    print('Connected by', addr)

    def calculateEnergy(self, values):
        return -(np.mean(values))

    def sender(self):
        """Process values and then send them all to processing-program.
        Read diffs[][], calculate and get ch[].
        """
        while 1:
            if self.fetch_ch_data():
                pos = []
                # transmission mode
                for i in range(self.r * self.c):
                    diff = self.diffs_p[i]
                    if diff > 0:
                        pos.append(diff / 50)   # 40make sense

                    else:
                        pos.append(diff / 50)
                # load mode
                for i in range(self.c * self.r):
                    diff = self.diffs[i]
                    if diff > 0:
                        pos.append(diff / 400000)   # 402528 make sense
                    else:
                        pos.append(diff / 400000)

                if self.conn:
                    try:
                        self.conn.send((' '.join(str(e) for e in pos) + '\n').encode())
                    except:
                        self.conn.close()
                        self.conn = None
                        print("Connectioon broken")

    def fetch_ch_data(self):
        """Read signal from arduino_port, Calibrate them by substracting base. So we get diffs[][]
        """
        self.fetch_arduino_data()   # realtime self.data
        if self.start_object_recog:  # R-pressed, start to proceess the thread
            self.calDiff()  # produce np.array: self.diffs[totalChannel], print data,base values out
            self.get_peak()  # get list: self.peak[r*c]
            self.calPosDiff()  # produce self.diffs_p[totalChannel]
        else:
            print(" :) Please press Reset on keyboard")
        if self.calibration:
            self.calibration = False
        if self.recalibration:
            self.calibration = True
            self.recalibration = False
        return True

    def fetch_arduino_data(self):
        start_time = time.time()
        result = self.arduino_port.readline().decode()
        # print("record took" + str(time.time() - start_time) + "s")
        result_arr = result.split(", ")
        result_arr_load = result_arr[0:144]
        result_arr_tran = result_arr[144:288]
        for i in range(self.r):
            for j in range(self.c):
                # fill in self.data from zero to Arduino port
                self.data[i * self.c + j][:-1] = self.data[i * self.c + j][1:]
                self.data[i * self.c + j][-1] = int(float(result_arr_load[i * self.c + j]))
                self.data_p[i * self.c + j][:-1] = self.data_p[i * self.c + j][1:]
                self.data_p[i * self.c + j][-1] = int(float(result_arr_tran[i * self.c + j]))
                if self.calibration:
                    self.base[i * self.c + j] = copy.deepcopy(self.data[i * self.c + j])
                    self.base_p[i * self.c + j] = copy.deepcopy(self.data_p[i * self.c + j])

    def processData(self, data):
        return np.median(data)  # To get mediam number of data

    def calDiff(self):
        """Calculate diffs[][]
        """
        for i in range(self.totalChannel):
            processed_data = self.processData(self.data[i])
            processed_base = self.processData(self.base[i])
            diff = processed_data - processed_base
            if diff < 0:
                self.diffs[i] = -diff
            else:
                self.diffs[i] = 0

    def get_peak(self):
        for i in range(self.r * self.c):
            if self.diffs[i] > self.peak[i]:    # list [0]*row*column
                self.peak[i] = self.diffs[i]

    def calPosDiff(self):
        """Calculate diffs_p[]
        with object on, diffs_p > 0
        """
        for i in range(self.totalChannel):
            processed_data_p = self.processData(self.data_p[i])
            processed_base_p = self.processData(self.base_p[i])
            diff = processed_data_p - processed_base_p
            if diff > 0:
                self.diffs_p[i] = diff
            else:
                self.diffs_p[i] = 0

    def reset(self):
        self.recalibration = True
        self.nonConductivePeak = np.ones((self.totalChannel)) * NON_CONDUCTIVE_PEAK
        self.conductivePeak = np.ones((self.totalChannel)) * CONDUCTIVE_PEAK
        self.capDecreasePeak = np.ones((self.totalChannel)) * CAP_DECREASE_PEAK
        self.diffs = np.zeros((self.totalChannel))
        # for ML
        self.peak = [0] * self.r * self.c
        self.ml_base = []
        self.mapping = []
        for i in range(self.r * self.c):
            self.mapping.append(i)  # self.mapping = [0,1,2,...,35]


    def start(self):
        t1 = threading.Thread(target=self.socket_handler, name="socket", args=())
        t1.start()
        self.sender()

    def save_peak(self):
        cwd = os.getcwd()  # current working dictionary
        filename = cwd + '\\coil\\' + 'peak' + '.csv'
        print("Filename: {}".format(filename))
        f = open(filename, mode='w+', encoding='utf-8')
        for i in range(self.r * self.c):
            f.write(str(self.peak[i]))
            f.write(", ")
        f.close()
        print('writen to file ' + filename)


if __name__ == '__main__':
    serialRead()
    fetchdata = FetchData()
    fetchdata.start()


