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
import matplotlib

# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
import math
import numpy as np
import os
import copy

from sklearn.neighbors import NearestNeighbors
# import feature_extraction8
from sklearn import preprocessing
from sklearn.externals import joblib

# import matplotlib

print(rcsetup.all_backends)
INTERNAL_MUX_MODE = 0
EXTERNAL_MUX_MODE = 1
WINDOW_SIZE = 10
MSP_CHANNEL = 4
CONDUCTIVE_THRESHOLD = 600000
NON_CONDUCTIVE_PEAK = 600000
CONDUCTIVE_PEAK = 5000000  # TO ++
CAP_DECREASE_PEAK = 100000
NON_NOISE_THRESHOLD = 5000
SINGLE_CAP_THRESHOLD = 50000
# Specific Port related to your device
ARDUINO_SERIAL_PORT = "COM4"
CHANELL = 4
SUPPORTED_ACTION_NUM = 50
ACTION_ENERGY_THRESHOLD = 30000
OBJECT_ENERGY_THRESHOLD = 1000
# categorization


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
    return serialNum, serialDict

class FetchData:
    '''
    Function: Build by set FDC and start sensor
    '''


    def __init__(self):
        # second

        self.mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mysocket.bind(('localhost', 5000))
        self.mysocket.listen(1)
        self.conn = None
        self.r = 8
        self.c = 8
        self.layer = 1
        self.recalibration = False
        self.calibration = True

        self.totalChannel = self.r * self.c

        self.data = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        self.base = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        # self.cur = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        # self.chg = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        self.index = 0
        self.action = np.zeros((SUPPORTED_ACTION_NUM))
        self.lastData = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        self.bias = np.zeros((SUPPORTED_ACTION_NUM, self.layer, self.totalChannel))

        self.arduino_port = serial.Serial(port=ARDUINO_SERIAL_PORT, baudrate=250000)
        self.arduino_port.reset_input_buffer()
        # time.sleep(1)
        self.start_object_recog = False;
        self.arduino_port.dsrdtr = 0
        # time.sleep(1)
        self.reset()
        print(self.arduino_port.readline())  # b'FDC SETTING\r\n'
        print(self.arduino_port.readline())  # b'Sensor Start\r\n'
        print('build successful')

    def socket_handler(self):
        while True:
            if self.conn:
                data = self.conn.recv(1024)
                if data:
                    d = data.decode("utf-8")
                    if d.strip() == "reset":
                        #s_beforeC = data
                        # self.getCurrent()   # fresh cur[k][i], to save data before changed
                        self.reset()    # fresh base, to save new base
                        # self.getChanged()   # get actual signal after placing objects
                        self.start_object_recog = True;

            else:
                self.conn, addr = self.mysocket.accept()
                if self.conn:
                    print('Connected by', addr)

    '''
    Process values and then send them all out by send()
    Read diffs[][], calculate and get ch[].
    '''
    def calculateEnergy(self, values):
        return -(np.mean(values))
    def sender(self):
        while 1:
            self.fetch_ch_data()
            ch = []
            rawch = []
            for k in range(self.layer):
                for i in range(self.r * self.c):
                    diff = self.diffs[k][i]
                    # print(i)
                    if diff < CONDUCTIVE_THRESHOLD:
                        ch.append(diff/self.nonConductivePeak[k][i])
                    else:
                        ch.append((-diff/self.conductivePeak[k][i]))

                # print (data[i]-base[i])
                # rawch.append(data[i])
            # for i in range(self.r * self.c):
            #     diff1 = self.processData(self.data[0][i]) - self.processData(self.base[0][i])
            #     diff2 = self.processData(self.data[1][i]) - self.processData(self.base[1][i])

            #     diff = abs(diff1)-abs(diff2)

            #     ch.append(diff/CONDUCTIVE_THRESHOLD)
            if self.conn:
                try:
                    self.conn.send((' '.join(str(e) for e in ch) + '\n').encode())
                except:
                    self.conn.close()
                    self.conn = None
                    print("Connectioon broken")
            # print("Base: " + ' '.join(str(e) for e in self.base[0]) + '\n')
            # print("Data: " + ' '.join(str(e) for e in self.data[0]) + '\n')
            # print("Low Peak: " + ' '.join(str(e) for e in self.nonConductivePeak) + '\n')
            # print("Upper Peak: " + ' '.join(str(e) for e in self.capDecreasePeak) + '\n')
            # print("Pre Deformed Data: " + ' '.join(str(e) for e in self.pre_deformed_data) + '\n')
           # print("Value: " + ' '.join(str(e) for e in ch) + '\n')
            # print("isConductive: " + ' '.join(str(e) for e in isConductive) +'\n')
            # self.conn.send(' '.join(str(e) for e in rawch) + '\n')
            # time.sleep(self.send_time)

    '''
    Read signal from arduino_port 
    Calibrate them by substracting base
    So we get diffs[][]
    '''

    def fetch_ch_data(self):

        # self.base = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.test = [0] * WINDOW_SIZE
        start_time = time.time()
        result = self.arduino_port.readline().decode()
        # print(("result"))
        # print((result))
        result_arr = result.split(", ")
#        print("result_arr")
#        print(result_arr)
        for i in range(self.r):
            for j in range(self.c):
                for k in range(self.layer):
                    # fill in self.data from zero to arduino_port
                    self.data[k][i * self.c + j][:-1] = self.data[k][i * self.c + j][1:]
                    self.data[k][i * self.c + j][-1] = int(result_arr[i * self.c * self.layer + j * self.layer + k])
                    if self.calibration:
                        self.base[k][i * self.c + j] = copy.deepcopy(self.data[k][i * self.c + j])
                        # self.action[self.index] = 1
        if self.start_object_recog:
            for k in range(self.layer):
                processed_data = np.array([self.processData(self.data[k][i]) for i in range(self.totalChannel)])
                processed_base = np.array([self.processData(self.base[k][i]) for i in range(self.totalChannel)])
                average_energy = self.calculateEnergy(processed_data - processed_base)
                print("AVERAGE ENERGY  " + str(average_energy))
                if average_energy > ACTION_ENERGY_THRESHOLD:
                    self.bias[self.index][k] = processed_data - processed_base
                    self.action[self.index] = 1
                    self.index += 1
                    self.base[k] = copy.deepcopy(self.data[k])
                    print("ACTION DOWN INDEX" + str(self.index))
                elif average_energy < -ACTION_ENERGY_THRESHOLD:
                    for i in range(self.index):
                        if self.action[self.index] == 1:
                            bias_energy = self.calculateEnergy(self.bias[self.index][k])
                            if abs(bias_energy - average_energy) < OBJECT_ENERGY_THRESHOLD:
                                self.action[i] = 0
                                # self.index += 1
                                self.base[k] = copy.deepcopy(self.data[k])
                                print("ACTION UP INDEX" + str(self.index))
                                break
            # for i in range(self.totalChannel):
            #     before_data = self.cur[k][i]
            #     changed_data = self.chg[k][i]
            #     self.bias[self.index][k][i] = changed_data - before_data
                # bias[index][k][i] = 0
        # print("Bias: ")
        # print(self.bias[self.index])

        self.lastData = copy.deepcopy(self.data)
        # print ("record took " +str(time.time()-start_time)+ "s")
        self.calDiff()  # now we get diff[][]
        # self.recgMaterial()  # TODO cancel #

        # self.adjustPeak()
        # self.scaleDiff()
#        self.cancelSingleCap2() # self.cancelSingleCap()
        if self.calibration:
            self.calibration = False
        if self.recalibration:
            self.calibration = True
            self.recalibration = False
        # time.sleep(0.01)
#        print("record took " + str(time.time() - start_time) + "s")

    def processData(self, data):
        return np.median(data)  # To get mediam number of data

    def adjustPeak(self):
        self.isTouch = False
        for k in range(self.layer):
            for i in range(self.totalChannel):
                diff = self.diffs[k][i]
                if diff < 0 and not self.recalibration:
                    if abs(diff) < CONDUCTIVE_THRESHOLD and abs(diff) > self.nonConductivePeak[k][i]:
                        self.nonConductivePeak[k][i] = abs(diff)
                    elif abs(diff) >= CONDUCTIVE_THRESHOLD:
                        if abs(diff) > self.conductivePeak[k][i]:
                            self.conductivePeak[k][i] = abs(diff)
                elif diff > 0 and abs(diff) > self.capDecreasePeak[k][i] and not self.recalibration:
                    self.capDecreasePeak[k][i] = abs(diff)
    '''
    get current signal
    '''

    def getCurrent(self):
        for k in range(self.layer):
            for i in range(self.totalChannel):
                processed_data = self.processData(self.data[k][i])
                processed_base = self.processData(self.base[k][i])
                diff = processed_data
                self.cur[k][i] = diff
                print("current diff: ")
                print(diff)
        # now we get current signal of the whole map

    def getChanged(self):
        for k in range(self.layer):
            for i in range(self.totalChannel):
                processed_data = self.processData(self.data[k][i])
                processed_base = self.processData(self.base[k][i])
                diff = processed_base
                self.chg[k][i] = diff
                print("changed diff: ")
                print(diff)
        # now we get current signal of the whole map after changing

    '''
    Adjust diffs[][]
    '''
    def calDiff(self):
        for k in range(self.layer):
            for i in range(self.totalChannel):
                processed_data = self.processData(self.data[k][i])
                processed_base = self.processData(self.base[k][i])
                diff = processed_data - processed_base
                if(diff > 0):
                    # self.base[k][i] = [processed_base]*WINDOW_SIZE
                    diff = 0
                else:
                    diff = abs(diff)
#                print("diff")
#                print(diff)
                self.diffs[k][i] = diff

    def returnCalDiff(self):
        for k in range(self.layer):
            for i in range(self.totalChannel):
                processed_data = self.processData(self.data[k][i])
                processed_base = self.processData(self.base[k][i])
                return processed_data, processed_base

    def recgMaterial(self):
        # To get diffs[][]
        max_metrix = max(max((self.diffs)))
#        print("Max of the metrix is ", max_metrix)
        useful_points = []
        cover_rate = 0.9
        for k in range(self.layer):
            for i in range(self.r):
                for j in range(self.c):
                    index = i * self.c + j
                    t = abs(self.diffs[k][index])
                    if (t > max_metrix * cover_rate):
                        useful_points.append(t)
        avr = np.mean(useful_points)  # avr
#        print("Average of this type is ", avr)

    def scaleDiff(self):
        for k in range(self.layer):
            for i in range(self.totalChannel):
                self.diffs[k][i] = self.diffs[k][i] / self.conductivePeak[k][i]

    '''
    Subtract abs(minimum) in column and raw for diffs[]
    '''

    def cancelSingleCap2(self):
        candidates = []
        rate = 0.9
        for k in range(self.layer):
            for i in range(self.r):
                row_caps = [(self.diffs[k][i * self.c + j]) for j in range(self.c)]
                maximum = max(row_caps)
                minimum = min(row_caps)

                for j in range(self.c):
                    index = i * self.c + j
                    # if(self.diffs[k][index]) < maximum * rate :
                    self.diffs[k][index] = ((self.diffs[k][index]) - minimum)
                        # self.diffs[k][index] = 0

            for j in range(self.c):
                col_caps = [(self.diffs[k][i * self.c + j]) for i in range(self.r)]
                col_caps_filter = list(filter(lambda c: c > NON_NOISE_THRESHOLD, col_caps))
                print("filter")
                print(col_caps_filter)
                if len(col_caps_filter) == 0:
                    continue
                minimum = min(col_caps_filter)
                maximum = max(col_caps_filter)

                for i in range(self.r):
                    index = i * self.c + j
                    # if (self.diffs[k][index]) < maximum * rate:
                    self.diffs[k][index] = ((self.diffs[k][index]) - minimum)
                        # self.diffs[k][index] = 0

    def cancelSingleCap(self):

        candidates = []
        for k in range(self.layer):
            for i in range(self.r):
                row_caps = [abs(self.diffs[k][i * self.c + j]) for j in range(self.c)]
                minimum = min(row_caps)

                for j in range(self.c):
                    index = i * self.c + j
                    self.diffs[k][index] = -(abs(self.diffs[k][index]) - minimum)
            for j in range(self.c):
                col_caps = [abs(self.diffs[k][i * self.c + j]) if abs(
                    self.diffs[k][i * self.c + j]) > NON_NOISE_THRESHOLD else 10000000 for i in range(self.r)]
                minimum = min(col_caps)
                if minimum == 10000000:
                    continue
                for i in range(self.r):
                    index = i * self.c + j
                    self.diffs[k][index] = -(abs(self.diffs[k][index]) - minimum)
        # single_caps = []
        #
        # for i in range(self.r):
        #
        # for candidate in candidates:
        #     k = candidate[0]
        #     i = candidate[1]
        #     j = candidate[2]
        #     index = i*self.c + j
        #     isDiffCap = True
        #     minimumY = 10000000
        #     for y in range(0, self.r):
        #         index2 = y*self.c + j
        #         if self.diffs[k][index2] > 0:
        #             continue
        #         if abs(self.diffs[k][index2]) < minimumY:
        #             minimumY = abs(self.diffs[k][index2])
        #
        #     if abs(self.diffs[k][index]) - minimumY < SINGLE_CAP_THRESHOLD or minimumY < NON_NOISE_THRESHOLD:
        #         if minimumY < NON_NOISE_THRESHOLD:
        #             minimumY = abs(self.diffs[k][index])
        #         isDiffCap = False
        #
        #     # minimum = 10000000
        #     for x in range(0, self.c):
        #         index2 = i*self.c + x
        #         if self.diffs[k][index2] > 0:
        #             continue
        #         if abs(self.diffs[k][index2]) < minimum:
        #              minimum = abs(self.diffs[k][index2])
        #     if abs(self.diffs[k][index]) - minimum < SINGLE_CAP_THRESHOLD or minimum < NON_NOISE_THRESHOLD:
        #         if minimum < NON_NOISE_THRESHOLD:
        #             minimum = abs(self.diffs[k][index])
        #         isDiffCap = False
        #
        #     if not isDiffCap:
        #         single_caps.append((k, i, j, minimum))
        #
        # for candidate in single_caps:
        #     k = candidate[0]
        #     i = candidate[1]
        #     j = candidate[2]
        #     minimum = candidate[3]
        #     index = i*self.c + j
        #     self.diffs[k][index] += minimum

    def reset(self):
        self.recalibration = True
        self.nonConductivePeak = np.ones((self.layer, self.totalChannel)) * NON_CONDUCTIVE_PEAK
        self.conductivePeak = np.ones((self.layer, self.totalChannel)) * CONDUCTIVE_PEAK
        self.capDecreasePeak = np.ones((self.layer, self.totalChannel)) * CAP_DECREASE_PEAK
        self.arduino_port.write("reset\n".encode())
        self.diffs = np.zeros((self.layer, self.totalChannel))

        # self.pre_deformed_data = np.ones((self.layer, self.totalChannel))

    def start(self):

        # plt.ion()
        # fig = plt.figure()

        # self.test = [0]*WINDOW_SIZE
        # ax = fig.add_subplot(111)
        # ax.autoscale(enable=True, axis='y', tight=True)
        # ax.set_xlim([0, 10])
        # ax.set_ylim([26100000, 26400000])
        # line1, = ax.plot(list(range(WINDOW_SIZE)), self.test, 'r-') # Returns a tuple of line objects, thus the comma

        # t = threading.Thread(target=self.sender, name="sender", args=())
        # t.daemon = True
        # t.start()
        t1 = threading.Thread(target=self.socket_handler, name="socket", args=())
        t1.daemon = True
        t1.start()
        self.sender()
        # for msp in self.msps:
        #     msp.start()
        # try:
        #     while True:
        #         print(self.test)
        #         line1.set_ydata(self.test)
        #         fig.canvas.draw()
        #         fig.canvas.flush_events()
        #         ax3.set_ylim(min(self.test)-100000, max(self.test)+100000)
        #         time.sleep(0.15)
        # except KeyboardInterrupt:
        #     exit()


if __name__ == '__main__':
    (serialRead())
    # MSPComm("/dev/cu.usbmodem14141", "Test")
    fetchdata = FetchData()
    fetchdata.start()
