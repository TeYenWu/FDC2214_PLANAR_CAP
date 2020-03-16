# _*_ coding: utf-8 _*_

import socket
import time
import threading
import serial.tools.list_ports
import serial
import numpy as np
import math
import os
import feature_extraction8
import copy
import cv2
import csv
import random
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing

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
TRANS_THRESHOLD = 5


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
        # "glass", "bowl", "lipstickTF", "avocado"
        self.object_set = ["glass", "bowl", "lipstickTF", "avocado", "airpods", "book", "awardCARD", "discover_card",
                           "dry_flower","wet_flower", "green", "grapefruit", "toshiba", "bowl+soup", "kiwifruit",
                           "handsoap", "glass+water", "salt", "candle", "cheese1"]
        self.trainNumber = 10
        self.trainCount = 0

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
                        self.reset()  # Set self.recalibration = True, self.diffs = [0]
                        self.start_object_recog = True
                    elif d.strip() == "log":    # L-key on keyboard is pressed
                        # print("             log")
                        self.fetch_ml_data(self.object_set[self.trainCount//self.trainNumber])
                    elif d.strip() == "drawback":   # drawback
                        self.drawback()

            else:
                self.conn, addr = self.mysocket.accept()
                if self.conn:
                    print('Connected by', addr)

    def calculateEnergy(self, values):
        return -(np.mean(values))

    def read_peak(self):
        """To get list: self.peak[]
        """
        cwd = os.getcwd()  # current working dictionary
        filename = cwd + '\\coil\\' + 'peak' + '.csv'
        f = open(filename, 'r')
        line = f.readline()
        items = line.strip('\n').split(',')
        for i in items:
            # print(i)
            self.peak.append(i)
        f.close()
        self.peak = self.peak[:144]  # remove empty space
        # print("list peak: {}".format(self.peak))
        # print("len of list peak: {}".format(len(self.peak)))

    def sender(self):
        """Process values and then send them all to processing-program.
        Read diffs[][], calculate and get ch[].
        """
        while 1:
            if self.fetch_ch_data():
                pos = []

                # transmission mode
                self.read_peak()
                for i in range(self.r * self.c):
                    diff = self.diffs_p[i]
                    if diff > 0:
                        pos.append(diff / 50)   # 40make sense
                    else:
                        pos.append(diff / 50)

                # load mode
                for i in range(self.c * self.r):
                    diff = float(self.diffs[i]) / float(self.peak[i])
                    # print("diffs[{}]:{}, peak[{}]:{}, division={}".format(i, self.diffs[i], i, self.peak[i], diff))
                    if diff > 0:
                        pos.append(diff)   # 402528 make sense
                    else:
                        pos.append(diff)

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
            self.fetch_realtime_metrix()    # list: self.ml_data; self.ml_base
            # self.calDiff()  # produce self.diffs[totalChannel], print data,base values out
            # self.calPosDiff()   # produce self.diffs_p[totalChannel]
        else:
            print(" :) Please press Reset on keyboard")
        if self.calibration:
            self.calibration = False
        if self.recalibration:
            self.calibration = True
            self.recalibration = False
        return True

    def fetch_ml_data(self, name):
        #name = input('type object name to log data above')
        #print(name)
        data_list = []
        # for obj in objs.names:
        print('target data: ' + name)
        data1, base1 = [], []
        self.read_peak()

        for j in range(len(self.ml_data)):
            if self.ml_data[j] > 0:
                data1.append(float(self.ml_data[j])/float(self.peak[j]))   # load mode
                # print("diffs[{}]:{}, peak[{}]:{}, division={}".format(i, self.diffs[i], i, self.peak[i], diff))
            else:
                data1.append(0)

            if self.ml_base[j] > 0:
                base1.append(self.ml_base[j])   # transmission mode
            else:
                base1.append(0)

        line = str(name) + ','
        for d in data1:
            line += str(d) + ','
        for b in base1:
            line += str(b) + ','
        line_feature = line + str(name)
        iterms = line_feature.strip('\n').split(',')
        coilNums = 144  # self.r * self.c
        load_diff = iterms[1:coilNums + 1]
        load_diff = list(map(float, load_diff))
        trans_diff = iterms[coilNums + 1:2 * coilNums + 1]
        trans_diff = list(map(float, trans_diff))
        trans_diff = [c / 50 for c in trans_diff]
        # extract the features: 23 types of~
        feature = feature_extraction8.feature_calculation(load_diff, trans_diff)
        # print("FEATURE: {}".format(feature))
        clf = joblib.load("RF2.model")
        scalar = joblib.load("scalar.save")
        item_feature = feature
        # print("value of item_feature: {}".format(item_feature))
        for k in range(len(item_feature)):
            item_feature[k] = float(item_feature[k])
        test_list = item_feature
        # print("Len of test_list is: {}".format(len(test_list)))  # 72 features in one try
        # print("value of test_list: {}".format(test_list))

        # 将数据改成ndarray格式
        x = np.array(test_list, dtype=np.float64)
        x = scalar.transform([x])
        x = np.nan_to_num(x)
        x = x.reshape(1, -1)
        prediction = clf.predict(x)
        print("prediction: {}".format(prediction))

        cwd = os.getcwd()  # current working dictionary
        ffilename = cwd + '\\userstudy\\' + 'c' + str(self.trainCount // self.trainNumber) + '.csv'
        # print("Filename: {}".format(filename))
        ff = open(ffilename, mode='a+', encoding='utf-8')
        ff.write(str(prediction) + '\n')
        ff.close()
        # print('writen {} to file {} '.format(prediction, ffilename))

        # LOG USER DATA MEANWHILE
        line += str(name) + '\n'
        data_list.append(line)

        cwd = os.getcwd()  # current working dictionary
        filename = cwd + '\\userstudy\\' + 'user' + str(self.trainCount//self.trainNumber) + '.csv'
        f = open(filename, mode='a+', encoding='utf-8')
        for line in data_list:
            f.write(line)
        f.close()
        # print('writen to file ' + filename)

        self.trainCount += 1
        if self.trainCount % self.trainNumber == 0:
            print("\n start data collection of the next object \n")


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

    def fetch_realtime_metrix(self):
        """From self.c*self.r*10 to self.c*self.r,
            so we get real time position metrix and energy metrix(self.c * self.r).
        :return:
        """
        processed_data = np.array([self.processData(self.data[i]) for i in range(self.totalChannel)])
        processed_base = np.array([self.processData(self.base[i]) for i in range(self.totalChannel)])
        processed_data_p = np.array([self.processData(self.data_p[i]) for i in range(self.totalChannel)])
        processed_base_p = np.array([self.processData(self.base_p[i]) for i in range(self.totalChannel)])

        # LOAD MODE
        self.ml_data = self.energy_metrix = processed_base - processed_data    # 16*16 np array. energy metrix [real time position] = energy
        self.ml_data = self.ml_data.tolist()
        for i in self.ml_data:
            if i < 0:
                i = 0

        # TRANSMISSION MODE
        self.ml_base = self.pos_metrix = processed_data_p - processed_base_p
        # 16*16 np array. pos_metrix [real time position] = energy
        # object down: pos_metrix > 0
        self.ml_base = self.ml_base.tolist()
        for i in self.ml_base:
            if i < 0:
                i = 0

        print("BEFORE CALI: max of transmission is:{}".format(np.amax(self.ml_base)))
        print("BEFORE CALI: max of load is:{}".format(np.amax(self.ml_data)))
        # print("\n")

        if np.amax(self.ml_base) <= 7 or np.amax(self.ml_data) < 10000:
            self.diffs = np.zeros((self.totalChannel))
            self.diffs_p = np.zeros((self.totalChannel))
            print("*************auto cali***************")
            self.ml_base = [0] * self.r * self.c
            self.ml_data = [0] * self.r * self.c
            # print("AFTER CALI: max of transmission is:{}".format(np.amax(self.ml_base)))
            # print("AFTER CALI: max of load is:{}".format(np.amax(self.ml_data)))
            print("\n")
            # self.recalibration = True
            self.start_object_recog = True

        else:
            for i in range(self.totalChannel):
                diff = self.processData(self.data[i]) - self.processData(self.base[i])
                if diff < 0:
                    self.diffs[i] = -diff  # TODO: recg >0? <0?
                else:
                    self.diffs[i] = 0

                diff_t = self.processData(self.data_p[i]) - self.processData(self.base_p[i])
                if diff_t > 0:
                    self.diffs_p[i] = diff_t
                else:
                    self.diffs_p[i] = 0




    def processData(self, data):
        return np.median(data)  # To get mediam number of data

    def adjustPeak(self):
        self.isTouch = False
        for i in range(self.totalChannel):
            diff = self.diffs[i]
            if diff < 0 and not self.recalibration:
                if abs(diff) < CONDUCTIVE_THRESHOLD and abs(diff) > self.nonConductivePeak[i]:
                    self.nonConductivePeak[i] = abs(diff)
                elif abs(diff) >= CONDUCTIVE_THRESHOLD:
                    if abs(diff) > self.conductivePeak[i]:
                        self.conductivePeak[i] = abs(diff)
            elif diff > 0 and abs(diff) > self.capDecreasePeak[i] and not self.recalibration:
                self.capDecreasePeak[i] = abs(diff)


    def getCurrent(self):
        """Gets current signal.
        """
        for i in range(self.totalChannel):
            processed_data = self.processData(self.data[i])
            processed_base = self.processData(self.base[i])
            diff = processed_data
            self.cur[i] = diff
            print("current diff: ")
            print(diff)
    # now we get current signal of the whole map

    def getChanged(self):
        for i in range(self.totalChannel):
            processed_data = self.processData(self.data[i])
            processed_base = self.processData(self.base[i])
            diff = processed_base
            self.chg[i] = diff
            print("changed diff: ")
            print(diff)
        # now we get current signal of the whole map after changing


    def returnCalDiff(self):
        for i in range(self.totalChannel):
            processed_data = self.processData(self.data[i])
            processed_base = self.processData(self.base[i])
            return processed_data, processed_base

    def recgMaterial(self):
        # To get diffs[][]
        max_metrix = max(max((self.diffs)))
        #        print("Max of the metrix is ", max_metrix)
        useful_points = []
        cover_rate = 0.9
        for i in range(self.r):
            for j in range(self.c):
                index = i * self.c + j
                t = abs(self.diffs[index])
                if (t > max_metrix * cover_rate):
                    useful_points.append(t)
        avr = np.mean(useful_points)  # avr

    def scaleDiff(self):
        for i in range(self.totalChannel):
            self.diffs[i] = self.diffs[i] / self.conductivePeak[i]

    def cancelSingleCap2(self):
        """Subtract abs(minimum) in column and raw for diffs[]
        """
        candidates = []
        rate = 0.7
        for i in range(self.r):
            row_caps = [(self.diffs[i * self.c + j]) for j in range(self.c)]
            maximum = max(row_caps)
            minimum = min(row_caps)

            for j in range(self.c):
                index = i*self.c + j
                diff = self.diffs[index]
                if abs(diff) > NON_NOISE_THRESHOLD and diff < 0:
                    candidates.append((i, j))
        single_caps = []
        for candidate in candidates:
            i = candidate[0]
            j = candidate[1]
            index = i*self.c + j
            isDiffCap = True
            minimumY = 10000000
            for y in range(0, self.r):
                index2 = y*self.c + j
                if self.diffs[index2] > 0:
                    continue
                if abs(self.diffs[index2]) < minimum:
                    minimumY = abs(self.diffs[index2])

            if abs(self.diffs[index]) - minimum < SINGLE_CAP_THRESHOLD or minimum < NON_NOISE_THRESHOLD:
                if minimumY < NON_NOISE_THRESHOLD:
                    minimumY = abs(self.diffs[index])
                isDiffCap = False

            minimumX = 10000000
            for x in range(0, self.c):
                index2 = i*self.c + x
                if self.diffs[index2] > 0:
                    continue
                if abs(self.diffs[index2]) < minimumX:
                     minimumX = abs(self.diffs[index2])
            if abs(self.diffs[index]) - minimumX < SINGLE_CAP_THRESHOLD or minimumX < NON_NOISE_THRESHOLD:
                if minimumX < NON_NOISE_THRESHOLD:
                    minimumX = abs(self.diffs[index])
                isDiffCap = False

            if not isDiffCap:
                single_caps.append((i, j, minimumY, minimumX))

        for candidate in single_caps:
            i = candidate[0]
            j = candidate[1]
            minimum = candidate[2]
            index = i*self.c + j
            self.diffs[index] += minimum

    def reset(self):
        self.recalibration = True
        # self.nonConductivePeak = np.ones((self.totalChannel)) * NON_CONDUCTIVE_PEAK
        # self.conductivePeak = np.ones((self.totalChannel)) * CONDUCTIVE_PEAK
        # self.capDecreasePeak = np.ones((self.totalChannel)) * CAP_DECREASE_PEAK
        self.diffs = np.zeros((self.totalChannel))
        self.diffs_p = np.zeros((self.totalChannel))
        # for ML
        self.peak = []
        self.ml_base = [0] * self.r * self.c
        self.ml_data = [0] * self.r * self.c
        self.mapping = []
        for i in range(self.r * self.c):
            self.mapping.append(i)  # self.mapping = [0,1,2,...,35]
        # self.pre_deformed_data = np.ones((self.totalChannel))

    def start(self):
        t1 = threading.Thread(target=self.socket_handler, name="socket", args=())
        t1.start()
        self.sender()

    def drawback(self):
        # delete data
        cwd = os.getcwd()  # current working dictionary
        filename = cwd + '\\userstudy\\' + 'c' + str(self.trainCount // self.trainNumber) + '.csv'
        readFile = open(filename)
        lines = readFile.readlines()
        readFile.close()

        w = open(filename, "w")
        w.writelines([item for item in lines[:-1]])
        w.close
        print("delete result SUCCESS", end=";")

        filename = cwd + '\\userstudy\\' + 'user' + str(self.trainCount // self.trainNumber) + '.csv'
        readFile = open(filename)
        lines = readFile.readlines()
        readFile.close()

        w = open(filename, "w")
        w.writelines([item for item in lines[:-1]])
        w.close
        self.trainCount -= 1
        print("delete data SUCCESS\n")


if __name__ == '__main__':
    serialRead()
    fetchdata = FetchData()
    fetchdata.start()
