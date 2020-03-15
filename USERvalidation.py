# -*- coding: utf-8 -*-
# 用于将收集的数据做成模型，然后进行预测。

import feature_extraction8
import os
import joblib
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
import numpy as np


TYPE_NUM = 20   # TODO: changed
OBJLOG = 100
if __name__ == '__main__':
    correct_num = 0

    obj = ["glass", "bowl", "lipstickTF", "avocado", "airpods", "book", "awardCARD", "discover_card",
                       "dry_flower",
                       "wet_flower", "green", "grapefruit", "toshiba", "bowl+soup", "kiwifruit", "handsoap",
                       "glass+water", "salt",
                       "candle", "cheese1"]
    cwd = os.getcwd()
    feature_file = cwd + '\\userstudy\\user\\' + 'features.csv'
    f_feature = open(feature_file, 'r')
    for j in range(TYPE_NUM*OBJLOG):  # scan all lines
        line = f_feature.readline()
            # print("line: {}".format(line))
        iterms = line.strip('\n').split(',')
        clf = joblib.load("RF2.model")
        scalar = joblib.load("scalar.save")
        item_feature = iterms
        # print("value of item_feature: {}".format(item_feature))
        for k in range(len(item_feature)):
            item_feature[k] = float(item_feature[k])
        test_list = item_feature
        x = np.array(test_list, dtype=np.float64)
        x = scalar.transform([x])
        x = np.nan_to_num(x)
        x = x.reshape(1, -1)
        prediction = clf.predict(x)
        prediction = np.array2string(prediction)
        if obj[j // OBJLOG] != prediction[2:-2]:
            # print("\nprediction FAILED:)expect:{}:)prediction:{}:)".format(obj[j // OBJLOG], prediction))
            user_num = (j % OBJLOG) // TYPE_NUM
            # obj_num = j % OBJLOG

            print("line({})     user:{}       put_time:{}       expect:{}       prediction:{}".format(j, user_num, j % 10, obj[j // OBJLOG], prediction))
        else:
            correct_num += 1

        cwd = os.getcwd()  # current working dictionary
        ffilename = cwd + '\\userstudy\\' + 'result' + '.csv'
        # print("Filename: {}".format(filename))
        ff = open(ffilename, mode='a+', encoding='utf-8')
        ff.write(str(prediction) + '\n')
        ff.close()

    accuracy = correct_num / (TYPE_NUM*OBJLOG)
    print("accuracy{}/{}=".format(correct_num, TYPE_NUM*OBJLOG), end="")
    print(accuracy)







