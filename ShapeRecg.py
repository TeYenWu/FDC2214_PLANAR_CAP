'''
Draw Polygon with OpenCV
'''

import cv2
import numpy as np

SUPPORTED_OBJECT_NUM = 2
MAX_DRAW_POINT = 20
AREA_PERCENT = 0.7

class ShapeRecg:
    def __init__(self):
        self.r = 100    # max of x-cor
        self.c = 100    # max of y-cor
        self.energy_metrix = np.zeros((SUPPORTED_OBJECT_NUM, self.r, self.c))

    '''
    Function: For each object in self.energy_metrix, drawShape(object_seq) calculates its equivalent position point and energy
    :param: object_seq refers to index of object in elf.energy_metrix[SUPPORTED_ACTION_NUM][][]
    :returns: coordinate refers to x_coordinate * 16 + y；center_energy
    '''
    def drawShape(self, object_seq):
        canvas = np.ones((CANVAS_WIDTH, CANVAS_WIDTH, 3), dtype="uint8")  # Initialize a blank canvas, set it's size, channel, bg=white
        canvas *= 255
        points = np.zeros(MAX_DRAW_POINT*2, np.int32)  # each point has 2 values: x and y.
        count = 0   # number of processed points
        # To get max energy
        max_energy = 0
        single_point_energy = []
        # Get max energy for the object
        for i in range(self.r):
            for j in range(self.c):
                if self.energy_metrix[object_seq][i][j] > max_energy:
                    max_energy = self.energy_metrix[object_seq][i][j]
                    area_threshold = AREA_PERCENT * max_energy

        # Produce a selection of points to draw soon
        for i in range(self.r):
            for j in range(self.c):
                # Draw the point on canvas only when it's close to max_energy, thus noise is away.
                if self.energy_metrix[object_seq][i][j] > 0 and self.energy_metrix[object_seq][i][j] > area_threshold:
                    points[count] = int(i) * 50
                    count += 1
                    points[count] = int(j) * 50
                    count += 1
                    single_point_energy.append(self.energy_metrix[object_seq][i][j])

        point_set = points[0:count]     # remove extra empty points
        point_shaped = point_set.reshape(-1, (count+1) // 2, 2)  # reshape it to Object_number * (x,y); RESHAPE TO N*1*2
        # print("points after reshape")
        # print(point_shaped)
        '''
        pts : 点集数组
        isClosed: 多边形是否闭合
        color: 颜色
        thickness: 线条宽度
        '''
        cv2.polylines(canvas, pts=[point_shaped], isClosed=True, color=(0, 0, 255), thickness=1)
        # Calculate position and energy for capacity_center
        energy_sum = 0
        x_coordinate_sum = 0
        y_coordinate_sum = 0
        x_coordinate_energy_sum = 0
        y_coordinate_energy_sum = 0
        for i in range(0, count, 2):
            energy_sum += single_point_energy[i//2]     # energy of the whole area
            x_coordinate_energy_sum += point_set[i] * single_point_energy[i//2]
            y_coordinate_energy_sum += point_set[i+1] * single_point_energy[i//2]
        x_center = x_coordinate_energy_sum/ energy_sum
        y_center = y_coordinate_energy_sum / energy_sum
        center_energy = energy_sum / (count//2)

        cv2.imshow("polylines", canvas)
        cv2.imwrite("polylines.png", canvas)
        cv2.waitKey(0)
        print('x_center is {}, y_center is {}, center_energy is {}'.format(x_center, y_center, center_energy))
        coordinate = x_center * self.r + y_center
        return coordinate, center_energy

if __name__ == '__main__':

    metrix = np.zeros((1,2,2))
    metrix[0][0][0] = 10
    metrix[0][0][1] = 2
    metrix[0][1][0] = 3
    metrix[0][1][1] = 4

    object_shape.fillMetric(metrix[0], r=2, c=2)
    object_shape.drawShape(0)
