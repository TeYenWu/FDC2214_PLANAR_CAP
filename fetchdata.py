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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.rcsetup as rcsetup
import math
import os
import copy
import cv2
from sklearn.neighbors import NearestNeighbors
# import feature_extraction8
from sklearn import preprocessing
from sklearn.externals import joblib
from painter import *
# For W & R CSV file
import csv
import random
import time

matplotlib.use('TkAgg')
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
SUPPORTED_ACTION_NUM = 100
ACTION_ENERGY_THRESHOLD = 10000
OBJECT_ENERGY_THRESHOLD = 1000
MAX_DRAW_POINT = 256    # self.c * self.r
AREA_PERCENT = 0.8
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800


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
    def __init__(self):
        # second
        self.mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mysocket.bind(('localhost', 5000))
        self.mysocket.listen(1)
        self.conn = None
        self.r = 16
        self.c = 16
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
        self.energy_metrix = np.zeros((self.r, self.c))

        self.position = np.zeros((SUPPORTED_ACTION_NUM))  # the current poition(X*self.r+Y) of object(index)
        self.unknownPositionEnergy = np.zeros((self.r * self.c))
        # self.move = np.zeros((SUPPORTED_ACTION_NUM))  # if the position stays unchanged, move = false, else true
        self.track = np.array(self.position, dtype=np.str)  # the position history of object(index)
        self.tempIndex = 0
        # (用了下中文:> )循环中的i不是self对象的类变量，不能作为getCurrentPosition()函数的参数，所以建立用来临时传参的变量self.tempIndex表示目前遍历到的物体的索引值以返回这个物体的currentPosition

        self.lastData = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        self.objectEnergy = np.zeros((SUPPORTED_ACTION_NUM, self.layer, self.totalChannel))
        self.min_energy = 0
        self.last_down_data = 0
        self.point_set = []
        self.mv_energy = 0
        self.last_mv_pos = 0
        self.v_time = 0

        self.arduino_port = serial.Serial(port=ARDUINO_SERIAL_PORT, baudrate=250000)
        self.arduino_port.reset_input_buffer()
        # time.sleep(1)
        self.start_object_recog = False
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
                        # s_beforeC = data
                        # self.getCurrent()   # fresh cur[k][i], to save data before changed
                        self.reset()  # fresh base, to save new base
                        # self.getChanged()   # get actual signal after placing objects
                        self.start_object_recog = True

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
            self.fetch_ch_data()
            ch = []
            rawch = []

            for k in range(self.layer):
                for i in range(self.r * self.c):
                    diff = self.diffs[k][i]
                    # print(i)
                    if diff < CONDUCTIVE_THRESHOLD:
                        ch.append(diff / self.nonConductivePeak[k][i])
                    else:
                        ch.append((-diff / self.conductivePeak[k][i]))

            ''' unlike ch(if diff[]>0 diff[]=0), zh is raw data - base
            zh = []
            for i in range(self.r * self.c):
                zh.append(self.energy_metrix[self.index][i//8][i % 8])
            '''

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


    def fetch_ch_data(self):
        """Read signal from arduino_port, Calibrate them by substracting base. So we get diffs[][]
        """
        #self.timingTrack()  # tracking the position history of object
        # self.base = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.test = [0] * WINDOW_SIZE
        start_time = time.time()
        result = self.arduino_port.readline().decode()
        # print(("result"))
        # print((result))
        result_arr = result.split(", ")
        # print("result_arr")
        # print(result_arr)  # no increase

        for i in range(self.r):
            for j in range(self.c):
                for k in range(self.layer):
                    # fill in self.data from zero to arduino_port
                    self.data[k][i * self.c + j][:-1] = self.data[k][i * self.c + j][1:]
                    #                    print("self.data " + str(self.data))  # no increase
                    self.data[k][i * self.c + j][-1] = int(result_arr[i * self.c * self.layer + j * self.layer + k])
                    if self.calibration:
                        self.base[k][i * self.c + j] = copy.deepcopy(self.data[k][i * self.c + j])

        if self.start_object_recog:
            for k in range(self.layer):
                processed_data = np.array([self.processData(self.data[k][i]) for i in range(self.totalChannel)])
                processed_base = np.array([self.processData(self.base[k][i]) for i in range(self.totalChannel)])
                self.energy_metrix = processed_base - processed_data  # 1*64,energy at the moment
                # for i in self.energy_metrix:
                    # print("Energy_metix: {}".format(self.energy_metrix))
                last_mv_pos = self.last_mv_pos
                # print("last_mv_pos: {}".format(self.last_mv_pos))

                cur_mv_pos = self.newest_move()     # last_mv_pos updated
                # print("Current_mv_pos: {}".format(cur_mv_pos))

                self.unknownPositionEnergy[cur_mv_pos] = self.mv_energy
                cur_obj_energy = self.mv_energy
                # print("unknownposition: ")
                # print(self.unknownPositionEnergy)

                #self.timingTrack()
                realtime_base = np.mean(processed_base)
                realtime_data = np.mean(processed_data)    # decrease with non-conductive
                real_time_energy = realtime_base - realtime_data

                with open('energy.csv', 'a') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    # headers
                    info = {
                        "v_time": self.v_time,
                        "v_base": realtime_base,
                        "v_data": realtime_data,
                        "v_energy": realtime_base - realtime_data
                    }
                    csv_writer.writerow(info)
                    # print(self.v_time, realtime_base, realtime_data, realtime_data)


                if self.index == 0:
                    self.last_mv_pos = 0

                # Object DOWN && not the noise
                if cur_obj_energy > ACTION_ENERGY_THRESHOLD and self.object_filter(last_mv_pos, cur_mv_pos):
                    # add self.action[index], self.objectEnergy[index]
                    self.objectEnergy[self.index][k] = cur_obj_energy
                    self.action[self.index] = cur_mv_pos

                    # update value of min_energy
                    if self.index == 0:
                        self.min_energy = cur_obj_energy
                    elif cur_obj_energy < self.min_energy:
                        self.min_energy = cur_obj_energy

                    # update base, index, last_down_data
                    self.base[k] = copy.deepcopy(self.data[k])
                    print("                                            ACTION DOWN INDEX " + str(self.index) + ' energy ' + str(cur_obj_energy))
                    p_list = [i * 10 for i in self.point_set]
                    self.point_poly_fill(p_list)
                    # self.calPosition()
                    self.track[self.index] = "track object[" + str(self.index) + "]:"
                    self.index += 1
                    self.last_down_data = realtime_data
                    self.last_mv_pos = cur_mv_pos

                elif ((realtime_data - self.last_down_data) - self.min_energy) > 0:  # Object UP
                    for i in range(self.index):
                        if self.action[i] > 0:
                            # print("action[index]>0 detected.")
                            # object_energy = self.calculateEnergy(self.objectEnergy[self.index][k])
                            object_energy = np.mean(self.objectEnergy[i][k])  # energy of object i
                            # print("energy of object  "+str(i)+" is  " + str(object_energy))
                            # OBJECT_ENERGY_THRESHOLD: # True when things REALLY go UP
                            if abs(abs(realtime_data - self.last_down_data) - abs(object_energy)) < abs(object_energy) * 2:
                                self.unknownPositionEnergy[int(self.action[i])] = 0
                                self.action[i] = 0
                                # self.index += 1
                                self.base[k] = copy.deepcopy(self.data[k])
                                print("                                            ACTION UP INDEX" + str(i))
                                self.last_mv_pos = 0
                                break

            # for i in range(self.totalChannel):
            #     before_data = self.cur[k][i]
            #     changed_data = self.chg[k][i]
            #     self.objectEnergy[self.index][k][i] = changed_data - before_data
            # bias[index][k][i] = 0
        # print("Bias: ")
        # print(self.objectEnergy[self.index])

        self.lastData = copy.deepcopy(self.data)
        # print ("record took " +str(time.time()-start_time)+ "s")
        self.calDiff()  # now we get diff[][]
        # self.recgMaterial()  # TODO cancel #

        # self.adjustPeak()
        # self.scaleDiff()
        # self.cancelSingleCap2() # self.cancelSingleCap()
        if self.calibration:
            self.calibration = False
        if self.recalibration:
            self.calibration = True
            self.recalibration = False
        # time.sleep(0.01)

        # print("record took " + str(time.time() - start_time) + "s")
        # self.point_poly_fill(self.point_set)

    def newest_move(self):
        """Log the last movement on the cloth

        When something new is placed on the cloth,
        select meaningful points with value and get center point, equivalent energy.

        :returns: current moving point energy
        """
        points = np.zeros(MAX_DRAW_POINT * 2, np.int32)  # each point has 2 values: x and y.
        # To get max energy on the cloth
        max_energy = 0
        single_point_energy = []
        area_threshold = 0
        for i in range(self.r):
            for j in range(self.c):
                temp = self.energy_metrix[i * self.r + j]
                if temp > max_energy:
                    max_energy = temp
                    area_threshold = AREA_PERCENT * max_energy

        count = 0  # number of processed points
        # Produce a selection of points to draw soon
        for i in range(self.r):
            for j in range(self.c):
                # Draw the point on canvas only when it's close to max_energy, thus noise is away.
                temp = self.energy_metrix[i * self.r + j]
                if temp > 0 and temp > area_threshold:
                    points[count] = int(i)
                    # print("Points[{}] = {}".format(count,points[count]))
                    count += 1
                    points[count] = int(j)
                    count += 1
                    single_point_energy.append(temp)

        if count == 0:  # No signal at the moment
            # print("No point generated.")
            return 0
        self.point_set = points[0: count]  # remove extra empty points, pass values of point_set to function drawShape
        # Calculate position and energy for capacity_center
        energy_sum = 0
        x_coordinate_energy_sum = 0
        y_coordinate_energy_sum = 0
        for i in range(0, count, 2):
            energy_sum += single_point_energy[i // 2]  # energy of the whole area
            x_coordinate_energy_sum += self.point_set[i] * single_point_energy[i // 2]
            y_coordinate_energy_sum += self.point_set[i + 1] * single_point_energy[i // 2]
        x_center = x_coordinate_energy_sum / energy_sum
        y_center = y_coordinate_energy_sum / energy_sum
        center_energy = energy_sum / (count // 2)
        #print('x_center is {}, y_center is {}, center_energy is {}'.format(x_center, y_center, center_energy))
        mv_cooridinate = int(x_center * self.r + y_center)
        self.mv_energy = center_energy
        self.last_mv_pos = mv_cooridinate
        return mv_cooridinate

    def object_filter(self, last_mv_pos, cur_mv_pos):
        t = abs(cur_mv_pos - last_mv_pos)
        if t < last_mv_pos // 24:
            # print("cur {} - last {} = {} Noise ignored.".format(cur_mv_pos, last_mv_pos, cur_mv_pos - last_mv_pos))
            return False    # noise
        else:
            # print("cur {} - last {} = {} Real object detected.".format(cur_mv_pos, last_mv_pos, cur_mv_pos - last_mv_pos))
            return True



    def drawShape(self):
        """For each object at the moment, draw its Shape.
        TODO: enable to recognize multiple objects with new algorithm
        """
        # Initialize a blank canvas, set it's size, channel, bg=white
        canvas = np.ones((CANVAS_WIDTH, CANVAS_WIDTH, 3), dtype="uint8")
        canvas *= 255
        # reshape to Object_number * (x,y); RESHAPE TO N*1*2
        point_shaped = self.point_set.reshape(-1, (count + 1) // 2, 2)
        for i in range(0, len(point_shaped)):
            point_shaped[i] = point_shaped[i] * 50
        # print("points after reshape")
        # print(point_shaped)
        '''Params for polylines
        pts : Set of points
        isClosed: polygon is closed or not
        color: red
        thickness: width of the line
        '''
        cv2.polylines(canvas, pts=[point_shaped], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.imshow("polylines", canvas)
        cv2.imwrite("polylines.png", canvas)
        # TODO: Too slow to run.
        # cv2.waitKey(0)

    def point_poly_fill(self, point_set):
        """Given points list, draw them on canvas with poly_fill algorithm.

        :param point_set: a list, containing x1,y1,x2,y2...xn,yn
        :param object_seq: which object should be painted on canvas
        :return: returns nothing. Just draw pic as response.
        """
        # Construct object list
        object_list = []
        for i in range(0, len(point_set), 2):
            found = False
            if len(object_list) == 0:  # First object
                object_list.append(Blob(point_set[i], point_set[i + 1]))
                found = True
            else:
                for obj in range(len(object_list)):
                    if object_list[obj].is_near(point_set[i], point_set[i + 1]):
                        object_list[obj].update_rec(point_set[i], point_set[i + 1])
                        found = True
                        # print_(object_list)
                        break
            if not found:
                object_list.append(Blob(point_set[i], point_set[i + 1]))
                found = True
                # print_(object_list)

        # Show object list
        if len(object_list) > 0:  # objects exist
            print("{} object(s) detected on the map.".format(len(object_list)))
            print_(object_list)
            polyfill(object_list[0].point_set)


    # every thread, check and record the position history of every object(from indexMin to indexMax)
    def timingTrack(self):
        # print("start tracking object")
        for i in range(self.index):
            # self.tempIndex = i
            if self.action[i] > 0:  # object(index) is down
                if self.getCurrentPosition(i) != self.position[i]:
                    # print("object[" + str(i) + "] position unchanged")
                    # self.move[i] = False  # position of the object(index) stay unchanged
                # else:
                    # self.move[i] = True  # position of the object(index) changed
                    self.position[i] = self.getCurrentPosition(i)
                    self.track[i] = self.track[i] + "  " + str(int(self.position[i]//self.r))+","+str(int(self.position[i]%self.c))  # str(self.position[i])
            if self.action[i]:
                print(self.track[i])

    def getCurrentPosition(self, index):
        for j in range(self.r * self.c):
            # print("unknownPositionEnergy[x*16+y=" + str(j) + "]=" + str(self.unknownPositionEnergy[j]))
            if 0.3*np.mean(self.objectEnergy[index][0]) < self.unknownPositionEnergy[j] < 3*np.mean(self.objectEnergy[index][0]):  ##########################别忘了改成阈值！
                print("unknownPositionEnergy[x*16+y="+str(j)+"]="+str(self.unknownPositionEnergy[j]))
                return j
        # return 200

    def processData(self, data):

        # print('np.median(data)')
        # print(np.median(data)) # no increase
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


    def getCurrent(self):
        """Gets current signal.
        """
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


    def calDiff(self):
        """Calculate diffs[][]
        """
        for k in range(self.layer):
            for i in range(self.totalChannel):
                processed_data = self.processData(self.data[k][i])
                processed_base = self.processData(self.base[k][i])
                diff = processed_data - processed_base
                if (diff > 0):
                    # self.base[k][i] = [processed_base]*WINDOW_SIZE
                    diff = 0
                else:
                    diff = abs(diff)
                #                print("diff")
                #                print(diff)
                self.diffs[k][i] = diff

    def calPosition(self):
        print("Diffs: ")
        for k in range(self.layer):
            for i in range(self.totalChannel):
                print(self.diffs[k][i])

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


    def cancelSingleCap2(self):
        """Subtract abs(minimum) in column and raw for diffs[]
        """
        candidates = []
        rate = 0.7
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
                # print("filter")
                # print(col_caps_filter)
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
    with open('energy.csv', 'w') as csv_file:
        fieldnames = ["v_time", "v_base", "v_data", "v_energy"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    serialRead()
    # MSPComm("/dev/cu.usbmodem14141", "Test")
    fetchdata = FetchData()
    fetchdata.start()