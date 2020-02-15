"""Polygon Scanning algorithm
class SingleLinkList: place linklist with class
PoliFill(image, polygon, color):tranfer by scanning
"""
import numpy as np
import matplotlib.pyplot as plt


class Node:
    # Define node
    def __init__(self, data):
        self._data = data
        self._next = None

    def get_data(self):
        return self._data

    def get_next(self):
        return self._next

    def set_data(self, ddata):
        self._data = ddata

    def set_next(self, nnext):
        self._next = nnext


class SingleLinkList:
    def __init__(self):
        # initialize list to be empty
        self._head = None
        self._size = 0

    def get_head(self):
        return self._head

    def is_empty(self):
        return self._head is None

    def append(self, data):
        # append in the rear
        temp = Node(data)
        if self._head is None:
            self._head = temp
        else:
            node = self._head
            while node.get_next() is not None:
                node = node.get_next()
            node.set_next(temp)
        self._size += 1

    def remove(self, data):
        # delete a node at the rear
        node = self._head
        prev = None
        while node is not None:
            if node.get_data() == data:
                if not prev:
                    # father_node is NONE
                    self._head = node.get_next()
                else:
                    prev.set_next(node.get_next())
                break
            else:
                prev = node
                node = node.get_next()
        self._size -= 1


def PoliFill(image, polygon, color):
    l = len(polygon)
    Ymax = 0
    Ymin = np.shape(image)[1]
    (width, height) = np.shape(image)
    # Calculate max and min edge
    for [x, y] in enumerate(polygon):
        if y[1] < Ymin:
            Ymin = y[1]
        if y[1] > Ymax:
            Ymax = y[1]

    # initialize and set up NET structure
    NET = []
    for i in range(height):
        NET.append(None)

    for i in range(Ymin, Ymax + 1):
        for j in range(0, l):
            if polygon[j][1] == i:
                # left_y > y0?
                if(polygon[(j-1+l) % l][1]) > polygon[j][1]:
                    [x1, y1] = polygon[(j-1+l) % l]
                    [x0, y0] = polygon[j]
                    delta_x = (x1-x0)/(y1-y0)
                    NET[i] = SingleLinkList()
                    NET[i].append([x0, delta_x, y1])

                # right_y > y0?
                if (polygon[(j+1+l) % l][1]) > polygon[j][1]:
                    [x1, y1] = polygon[(j + 1 + l) % l]
                    [x0, y0] = polygon[j]
                    delta_x = (x1 - x0) / (y1 - y0)
                    if NET[i] is not None:
                        NET[i].append([x0, delta_x, y1])
                    else:
                        NET[i] = SingleLinkList()
                        NET[i].append([x0, delta_x, y1])


    # set up active edge table
    AET = SingleLinkList()
    for y in range(Ymin, Ymax+1):
        # update start_x
        if not AET.is_empty():
            node = AET.get_head()
            while True:
                [start_x, delta_x, ymax] = node.get_data()
                start_x += delta_x
                node.set_data([start_x, delta_x, ymax])
                node = node.get_next()
                if node is None:
                    break

        # fill in
        if not AET.is_empty():
            node = AET.get_head()
            x_list = []
            # get all x_coordinates of cross points
            while True:
                [start_x, _, _] = node.get_data()
                x_list.append(start_x)
                node = node.get_next()
                if node is None:
                    break

            # sort
            x_list.sort()
            # pair 2 points and fill
            for i in range(0, len(x_list), 2):
                x1 = x_list[i]
                x2 = x_list[i+1]
                for pixel in range(int(x1), int(x2)+1):
                    image[y][pixel] = color

        if not AET.is_empty():
            # delete inactive edges
            node = AET.get_head()
            while True:
                [start_x, delta_x, ymax] = node.get_data()
                if ymax == y:
                    AET.remove([start_x, delta_x, ymax])
                node = node.get_next()
                if node is None:
                    break

        # add active edge
        if NET[y] is not None:
            node = NET[y].get_head()
            while True:
                AET.append(node.get_data())
                node = node.get_next()
                if node is None:
                    break

