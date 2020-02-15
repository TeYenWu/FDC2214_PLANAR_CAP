#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Recognize objects with rectangle.
TODO: improve to polygon, instead of rectangle.
'''

import cv2
import numpy as np
from poly_fill import *


distThreshold = 50  # adjust with electrode size and object size
CANVAS_WIDTH = 400


class Blob:
    def __init__(self, left_x, top_y):
        self.minx = left_x
        self.miny = top_y
        self.maxx = left_x
        self.maxy = top_y
        self.point_set = []
        self.point_set.append(top_y)
        self.point_set.append(left_x)

    # def show(self):
    #   draw

    def update_rec(self, new_x, new_y):
        """Once meets new point, update rectangle.

        :param new_x:
        :param new_y:
        :return:
        """
        self.minx = min(self.minx, new_x)
        self.miny = min(self.miny, new_y)
        self.maxx = max(self.maxx, new_x)
        self.maxy = max(self.maxy, new_y)
        self.point_set.append(new_y)
        self.point_set.append(new_x)  # add to rear

    def size(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def dist(self, x, y, xx, yy):
        return (x - xx) ** 2 + (y - yy) ** 2

    def is_near(self, new_x, new_y):
        center_x = (self.minx + self.maxx) / 2
        center_y = (self.miny + self.maxy) / 2
        d = self.dist(center_x, center_y, new_x, new_y)
        if d < distThreshold ** 2:
            return True
        else:
            return False


def print_(object_list):
    for i in range(len(object_list)):
        object_list[i].point_set = object_list[i].point_set[::-1]
        print("object{}: {}".format(i + 1, object_list[i].point_set))


def open_cv_draw_shape(point_list):
    """For each object at the moment, draw its Shape.
    TODO: enable to recognize multiple objects with new algorithm
    """
    count = len(point_list)
    point_list = np.asarray(point_list)
    # Initialize a blank canvas, set it's size, channel, bg=white
    canvas = np.ones((CANVAS_WIDTH, CANVAS_WIDTH, 3), dtype="uint8")
    canvas *= 255
    # reshape to Object_number * (x,y); RESHAPE TO N*1*2

    point_shaped = point_list.reshape(-1, count // 2, 2)
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
    cv2.polylines(canvas, pts=[point_shaped], isClosed=False, color=(0, 0, 255), thickness=1)
    cv2.imshow("polylines", canvas)
    cv2.imwrite("polylines.png", canvas)
    # TODO: Too slow to run.
    # cv2.waitKey(0)


def polyfill(object_point_list):
    image = np.ones([160, 160])
    plt.xlim(0, 160)  # x_coordinate ranges from 0 to 16
    plt.ylim(160, 0)



    # object1 = [150, 10, 0, 10, 10, 0]
    poly = []
    for i in range(0, len(object_point_list), 2):
        poly.append([object_point_list[i], object_point_list[i+1]])
    print("poly: {}.".format(poly))

    PoliFill(image, poly, False)
    plt.imshow(image, plt.cm.magma)  # display a 2D image in magma(color mode)
    plt.show()

