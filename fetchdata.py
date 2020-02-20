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
import time

# matplotlib.use('TkAgg')
# print(rcsetup.all_backend s)
INTERNAL_MUX_MODE = 0
EXTERNAL_MUX_MODE = 1
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
MAX_DRAW_POINT = 256    # self.c * self.r
AREA_PERCENT = 0.7
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
    print("Serial Read Success.")
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
        self.recalibration = False
        self.calibration = True
        self.totalChannel = self.r * self.c
        self.data = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.base = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.data_p = np.zeros((self.totalChannel, WINDOW_SIZE))
        self.base_p = np.zeros((self.totalChannel, WINDOW_SIZE))

        # self.cur = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        # self.chg = np.zeros((self.layer, self.totalChannel, WINDOW_SIZE))
        self.index = 0
        self.action = np.zeros((SUPPORTED_ACTION_NUM))
        self.energy_metrix = np.zeros((self.r, self.c))
        self.pos_metrix = np.zeros((self.r, self.c))

        self.position = np.zeros((SUPPORTED_ACTION_NUM))  # the current poition(X*self.r+Y) of object(index)
        self.unknownPositionEnergy = np.zeros((self.r * self.c))
        # self.move = np.zeros((SUPPORTED_ACTION_NUM))  # if the position stays unchanged, move = false, else true
        self.track = np.array(self.position, dtype=np.str)  # the position history of object(index)
        self.tempIndex = 0
        # (用了下中文:> )循环中的i不是self对象的类变量，不能作为getCurrentPosition()函数的参数，所以建立用来临时传参的变量self.tempIndex表示目前遍历到的物体的索引值以返回这个物体的currentPosition

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
            if self.fetch_ch_data():
                pos = []
                for i in range(self.r * self.c):
                    diff = self.diffs_p[i]
                    if diff > 0:
                        pos.append(diff / 40)
                    else:
                        pos.append(diff / 40)
                for i in range(self.c * self.r):
                    diff = self.diffs[i]
                    if diff > 0:
                        pos.append(diff / 402528)
                    else:
                        pos.append(diff / 402528)

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
        # print("fetchData")
        result = self.arduino_port.readline().decode()
        result_arr = result.split(", ")
        result_arr_load = result_arr[0:64]
        result_arr_tran = result_arr[64:128]

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

        if self.start_object_recog: # Press R on key board
            # from self.c*self.r*10 to self.c*self.r, so we get real time position metrix and energy metrix(self.c * self.r).
            processed_data = np.array([self.processData(self.data[i]) for i in range(self.totalChannel)])
            processed_base = np.array([self.processData(self.base[i]) for i in range(self.totalChannel)])
            processed_data_p = np.array([self.processData(self.data_p[i]) for i in range(self.totalChannel)])
            processed_base_p = np.array([self.processData(self.base_p[i]) for i in range(self.totalChannel)])

            self.energy_metrix = processed_base - processed_data    # 16*16 np array. energy metrix [real time position] = energy
            self.pos_metrix = processed_data_p - processed_base_p   # 16*16 np array. pos_metrix [real time position] = energy
                                                                    # object down: pos_metrix > 0
            last_mv_pos = self.last_mv_pos  # float=0, real time coordinate(self.r * x + y)
            self.calDiff()  # produce self.diffs[totalChannel]
            self.calPosDiff()   # produce self.diffs_p[totalChannel]

            t = self.newest_move()  # attention: last_mv_pos updated; mv_energy updated.
            if t != -1:
                cur_mv_pos = t  # equivalent x_center + self.r * y_center
            else:   # no noise nor object currently
                return False

            # TODO: currently, we assert here's only one object on map
                    # So self.mv_energy refers to object energy
            #self.unknownPositionEnergy[cur_mv_pos] = self.mv_energy  # real-time center energy.

            if self.object_filter(last_mv_pos, cur_mv_pos):    # obj DOWN
                self.update_(cur_mv_pos)    # update min_energy, log ID-energy-position

        else:
            print(" :) Please press Reset on keyboard")



        #self.lastData = copy.deepcopy(self.data)
        # print ("record took " +str(time.time()-start_time)+ "s")
        #self.calDiff()  # now we get diff[][]


        if self.calibration:
            self.calibration = False
        if self.recalibration:
            self.calibration = True
            self.recalibration = False
        return True

    def update_(self, cur_mv_pos):
        """Update status after recognized new object

        Update:
                self.objectEnergy[self.index] = self.mv_energy
                    Function: log ID - energy
                self.action[self.index] = cur_mv_pos
                    Function: log ID - position
                float self.min_energy
                    Function: log min( object energy )
                np.array(self.r, self.c) self.base = self.data
                    Function: update load-mode-base
                self.index += 1
                self.last_mv_pos = cur_mv_pos
        :return:
        """
        # log ID-energy, ID-position
        self.objectEnergy[self.index] = self.mv_energy  # TODO: self.mv_energy is avr(map energy), not object energy actually
        self.action[self.index] = cur_mv_pos    # TODO: cur_mv_pos is change point center on map, not object actually

        # update self.min_energy
        if self.index == 0:
            self.min_energy = self.mv_energy
        elif self.mv_energy < self.min_energy:
            self.min_energy = self.mv_energy

        # update energy-base
        self.base = copy.deepcopy(self.data)
        print("                                            ACTION DOWN INDEX " + str(self.index) + ' energy ' + str(
            self.mv_energy))

        # TODO: recognize object from points
        #p_list = [i * 10 for i in self.point_set]
        #self.point_poly_fill(p_list)


        # TODO: track
        # self.track[self.index] = "track object[" + str(self.index) + "]:"

        # update index, last_down_data
        self.index += 1
        # self.last_down_data = realtime_data   # TODO: what's the use?
        self.last_mv_pos = cur_mv_pos


    def newest_move(self):
        """Calculate whole image and get equivalent position & energy

        Scan (np[self.r*self.c])self.diffs_p to get need-to-draw points set
                    save in self.point_set
                            Type: list
                            Contains: x1, y1, x2, y2, ...
        From all need-to-draw points, calculate their
                    x_center
                            Type: float
                    y_center
                            Type: float
                    self.mv_energy
                            Type: float
                            Meaning: center energy
        Update self.last_mv_pos
                    Type: float
                    Value: self.r * y + x

        :returns: self.r * y + x
                    Type: float
        """
        points = np.zeros(MAX_DRAW_POINT * 2, np.int32)  # (np list)each point has 2 values: x and y.
        # To get max energy of position on the cloth
        single_point_energy = []    # (list) load mode energy of useful positions, in seq

        max_energy = max(self.diffs_p)
        area_threshold = AREA_PERCENT * max_energy
        count = 0  # number of processed points

        # fetch need-to-draw points set
        for i in range(self.r):
            for j in range(self.c):
                # Draw the point on canvas only when it's close to max_energy, thus noise is away.
                temp = self.diffs_p[i * self.r + j]
                if temp > 0 and temp > area_threshold:
                    points[count] = int(j)  # x
                    # print("Points[{}] = {}".format(count,points[count]))
                    count += 1
                    points[count] = int(i)  # y
                    count += 1
                    temp_point_energy = self.diffs[i * self.r + j]
                    # print("Temp point energy: {}".format(temp_point_energy))
                    single_point_energy.append(temp_point_energy)

        if count == 0:  # No useful posint at the moment
            print("No useful point generated.")
            return 0
        self.point_set = points[0: count]  # (list)remove extra empty points, pass values of point_set to function drawShape
        # to draw shape, just use self.point_set

        # Calculate position and energy for capacity_center
        energy_sum = 0
        x_coordinate_energy_sum = 0
        y_coordinate_energy_sum = 0
        for i in range(0, count, 2):
            energy_sum += single_point_energy[i // 2]  # energy of the whole area
            x_coordinate_energy_sum += self.point_set[i] * single_point_energy[i // 2]
            y_coordinate_energy_sum += self.point_set[i + 1] * single_point_energy[i // 2]
        # print("energy_sum: {}".format(energy_sum))
        if energy_sum == 0:
            return -1

        x_center = x_coordinate_energy_sum / energy_sum
        print("x_center: {}".format(x_center + 1))
        y_center = y_coordinate_energy_sum / energy_sum
        print("y_center: {}".format(y_center + 1))
        center_energy = energy_sum / (count // 2)
        #print('x_center is {}, y_center is {}, center_energy is {}'.format(x_center, y_center, center_energy))
        mv_cooridinate = int(y_center * self.r + x_center)
        self.mv_energy = center_energy  # (float) equivalent energy of one point
        print("center point energy: {}".format(center_energy))
        self.last_mv_pos = mv_cooridinate
        return mv_cooridinate

    def object_filter(self, last_mv_pos, cur_mv_pos):
        """Only when position and energy changed at the same time, return True

        :param last_mv_pos:
        :param cur_mv_pos:
        :return: True / False
        """
        map_energy = np.mean(self.energy_metrix)    # float, avr energy of real-time map
        t = abs(cur_mv_pos - last_mv_pos)

        if t < 1 or t < last_mv_pos / self.r:
            print("cur {} - last {} = {} Noise ignored.".format(cur_mv_pos, last_mv_pos, cur_mv_pos - last_mv_pos))
            return False    # noise
        elif map_energy > ACTION_ENERGY_THRESHOLD:
            print("cur {} - last {} = {} Real object detected.".format(cur_mv_pos, last_mv_pos, cur_mv_pos - last_mv_pos))
            return True
        else:
            return False


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


    def calDiff(self):
        """Calculate diffs[][]
        """
        for i in range(self.totalChannel):
            processed_data = self.processData(self.data[i])
            processed_base = self.processData(self.base[i])
            diff = processed_data - processed_base
            # if (diff > 0):
            #     # self.base[i] = [processed_base]*WINDOW_SIZE
            #     diff = 0
            # else:
            #     diff = abs(diff)
            self.diffs[i] = abs(diff)   # TODO: recg >0? <0?
        # print("diff")
        # print(self.diffs)

    def calPosDiff(self):
        """Calculate diffs_p[]

        with object on, diffs_p > 0
        """
        for i in range(self.totalChannel):
            processed_data_p = self.processData(self.data_p[i])
            processed_base_p = self.processData(self.base_p[i])
            diff = processed_data_p - processed_base_p
            self.diffs_p[i] = diff
        # print("diffs_p")
        # print(self.diffs_p)
        '''
        print("Check value of transmission mode data&base.")
        for i in range(self.r):
            for j in range(self.c):
                print(str(self.processData(self.data_p[i*self.r + j])) + "-" + str(self.processData(self.base_p[i*self.r + j])), end="  ")
            print("\n")
        '''

    # def calPosition(self):
    #     """Print diffs
    #
    #     :return:
    #     """
    #     print("Diffs: ")
    #     for i in range(self.totalChannel):
    #         print(self.diffs[i])

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

    #        print("Average of this type is ", avr)

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
        self.nonConductivePeak = np.ones((self.totalChannel)) * NON_CONDUCTIVE_PEAK
        self.conductivePeak = np.ones((self.totalChannel)) * CONDUCTIVE_PEAK
        self.capDecreasePeak = np.ones((self.totalChannel)) * CAP_DECREASE_PEAK
        self.diffs = np.zeros((self.totalChannel))
        # self.pre_deformed_data = np.ones((self.totalChannel))

    def start(self):
        t1 = threading.Thread(target=self.socket_handler, name="socket", args=())
        t1.daemon = True
        t1.start()
        self.sender()


if __name__ == '__main__':
    print(serialRead())
    # MSPComm("/dev/cu.usbmodem14141", "Test")
    fetchdata = FetchData()
    fetchdata.start()
