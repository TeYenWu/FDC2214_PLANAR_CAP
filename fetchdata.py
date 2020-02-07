# _*_ coding: utf-8 _*_

#3/30

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
CONDUCTIVE_THRESHOLD = 20000
NON_CONDUCTIVE_PEAK = 20000
CONDUCTIVE_PEAK = 150000
CAP_DECREASE_PEAK = 100000
NON_NOISE_THRESHOLD = 10000
SINGLE_CAP_THRESHOLD = 50000
ARDUINO_SERIAL_PORT = "COM4"
CHANELL = 4

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

        self.totalChannel =  self.r * self.c

        self.data = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        self.base = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))

        self.arduino_port = serial.Serial(port=ARDUINO_SERIAL_PORT, baudrate=250000)
        self.arduino_port.reset_input_buffer()
        # time.sleep(1)
        self.arduino_port.dsrdtr = 0
        # time.sleep(1)
        self.reset()
        print(self.arduino_port.readline())
        print(self.arduino_port.readline())
        print ('build successful')


    def socket_handler(self):
        while True:
            if self.conn:
                data = self.conn.recv(1024)
                if data:
                    d = data.decode("utf-8") 
                    if d.strip() == "reset":
                        self.reset()
            else:
                self.conn, addr = self.mysocket.accept()
                if self.conn:
                    print('Connected by', addr)

    def sender(self):
        while 1:
            self.fetch_ch_data()
            ch = []
            rawch = []
            for k in range(self.layer):       
                for i in range(self.r * self.c):
                    diff = self.diffs[k][i]
                    # print(i)
                    if diff < 0:
                        if abs(diff) < CONDUCTIVE_THRESHOLD:
                            ch.append(diff/self.nonConductivePeak[k][i])
                        else:
                            ch.append(-diff/self.conductivePeak[k][i])
                    else:
                        ch.append(-diff/CAP_DECREASE_PEAK)
                        # ch.append(0)

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
            print("Value: " + ' '.join(str(e) for e in ch) + '\n')
            # print("isConductive: " + ' '.join(str(e) for e in isConductive) +'\n')
            # self.conn.send(' '.join(str(e) for e in rawch) + '\n')
            # time.sleep(self.send_time)


    def fetch_ch_data(self):
        
        # self.base = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.test = [0]*WINDOW_SIZE
        start_time = time.time()
        result = self.arduino_port.readline().decode()
        # print(("result"))
        # print((result))
        result_arr = result.split(", ")
        # print(len(result_arr))
        for i in range(self.r):
            for j in range(self.c):
                for k in range(self.layer):
                    self.data[k][i*self.c + j][:-1] = self.data[k][i*self.c + j][1:]
                    self.data[k][i*self.c + j][-1] = int(result_arr[i*self.c * self.layer + j*self.layer+k])
                    if self.calibration:
                        self.base[k][i*self.c + j] = copy.deepcopy(self.data[k][i*self.c + j])

        # print ("record took " +str(time.time()-start_time)+ "s")
        self.calDiff()
        # self.adjustPeak()
        # self.scaleDiff()
        self.cancelSingleCap()
        if self.calibration:
            self.calibration = False   
        if self.recalibration:
            self.calibration = True  
            self.recalibration = False
        # time.sleep(0.01)
        print ("record took " +str(time.time()-start_time)+ "s")

    def processData(self, data):
        return np.median(data)

    def adjustPeak(self):
        self.isTouch = False
        for k in range(self.layer):                
            for i in range(self.totalChannel):
                diff = self.diffs[k][i]
                if diff < 0 and not self.recalibration:
                    if abs(diff) < CONDUCTIVE_THRESHOLD and abs(diff) > self.nonConductivePeak[k][i] :
                        self.nonConductivePeak[k][i] = abs(diff)
                    elif abs(diff) >= CONDUCTIVE_THRESHOLD:
                        if abs(diff) > self.conductivePeak[k][i]:
                            self.conductivePeak[k][i]= abs(diff)
                elif diff > 0 and abs(diff) > self.capDecreasePeak[k][i] and not self.recalibration:
                    self.capDecreasePeak[k][i] = abs(diff)

    def calDiff(self):
        for k in range(self.layer):                
            for i in range(self.totalChannel):
                processed_data = self.processData(self.data[k][i])
                processed_base = self.processData(self.base[k][i])
                diff = processed_data-processed_base
                self.diffs[k][i] = diff

    def scaleDiff(self):
        for k in range(self.layer):
            for i in range(self.totalChannel):
                self.diffs[k][i] = self.diffs[k][i]/self.conductivePeak[k][i]

    def cancelSingleCap(self):
        
        candidates = []
        for k in range(self.layer):
            for i in range(self.r):
                row_caps = [abs(self.diffs[k][i*self.c+j])  for j in range(self.c)]
                minimum = min(row_caps)

                for j in range(self.c):
                    index = i*self.c + j
                    self.diffs[k][index] = -(abs(self.diffs[k][index]) - minimum)
            for j in range(self.c):
                col_caps = [abs(self.diffs[k][i*self.c+j]) if abs(self.diffs[k][i*self.c+j]) > NON_NOISE_THRESHOLD else 10000000 for i in range(self.r)]
                minimum = min(col_caps)
                if minimum == 10000000:
                    continue
                for i in range(self.r):
                    index = i*self.c + j
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


