# -*- coding: utf-8 -*-
# 用于将收集的数据做成模型，然后进行预测。

import feature_extraction8
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import random

TYPE_NUM = 3    # TODO: changed
if __name__ == '__main__':
    print('calculate the features and store.')

    objectNums = TYPE_NUM
    trainDataNums = 10  # (line)number of samples for each object, origin: 30；# TODO: changed
    coilNums = 64   # self.r * self.c
    train_file_num = TYPE_NUM
    cwd = os.getcwd()

    train_list = []  # save every samples' features here
    label_list = []  # save every samples' label here
    train_type = []

    # for i in range(1,58):  # only use some kind of the objects to train and test.
    #     if i in(1,7,13,38,52,10,26,27,28,57,39,56,48,53,17,18,19,20,23):
    #         train_type.append(i)
    for i in range(0, train_file_num):
        train_type.append(i)    # train_type: [0,1,2, ... ]

    count = 0

    print("length of train type: "+str(len(train_type)))
    for i in train_type:    # scan all files
        count += 1

        # read the dataset of one object.
        filename = 'coil\\data\\c'+str(i)+'.csv'
        f = open(filename, 'r')
        for j in range(trainDataNums):  # scan all lines
            # data = []
            # base = []
            load_diff = []  # data
            trans_diff = []  # base

            line = f.readline()
            iterms = line.strip('\n').split(',')
            # print("iterms[0]: {}, iterms[1]: {}".format(iterms[0], iterms[1]))

            load_diff = iterms[1:coilNums+1]
            trans_diff = iterms[coilNums+1:2*coilNums+1]

            # extract the features: 23 types of~
            feature = feature_extraction8.feature_calculation(load_diff, trans_diff, i, j)   # i-which file; j-which line
            # print("feature: {}".format(feature))

            train_list.append(feature)  # one feature for each line
            label_list.append(i)
        print("i've added the " + str(count) + "th object")

        f.close()

    # save the features for each sample and save them label in lavel.csv with same order.
    cwd = os.getcwd()  # current working dictionary
    feature_file = cwd + '\\coil\\data\\' + 'features.csv'
    f_feature = open(feature_file, 'w')
    for i in train_list:
        current_line = ''
        for j in i:  # each feature
            current_line += str(j)+','
        current_line = current_line[:-1]
        current_line += '\n'
        f_feature.write(current_line)
    f_feature.close()

    label_file = cwd + '\\coil\\data\\' + 'label.csv'
    f_label = open(label_file, 'w')  # cover write
    current_line = ''
    for i in label_list:
        current_line += str(i)+','
    current_line = current_line[:-1]
    current_line += '\n'
    f_label.write(current_line)
    f_label.close()


