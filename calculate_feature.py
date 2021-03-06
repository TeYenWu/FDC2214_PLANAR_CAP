# -*- coding: utf-8 -*-
# 用于将收集的数据做成模型，然后进行预测。

import feature_extraction8
import os


TYPE_NUM = 20   # TODO: changed
if __name__ == '__main__':
    print('calculate the features and store.')

    objectNums = TYPE_NUM
    trainDataNums = 50  # (line)number of samples for each object, origin: 30；# TODO: changed
    coilNums = 144   # self.r * self.c
    train_file_num = TYPE_NUM
    cwd = os.getcwd()

    train_list = []  # save every samples' features here
    label_list = []  # save every samples' label here
    train_type = []

    for i in range(0, train_file_num):
        train_type.append(i)    # train_type: [0,1,2, ... ]
    count = 0
    print("length of train type: "+str(len(train_type)))
    obj = ["glass", "bowl", "lipstickTF", "avocado", "airpods", "book", "awardCARD", "discover_card", "dry_flower", "wet_flower", "green",  "grapefruit", "toshiba", "bowl+soup", "kiwifruit", "handsoap", "glass+water", "salt", "candle", "cheese1"]
    # obj = ["bowl", "glass+none", "glass+water", "eyeshadow", "toshiba", "clay", "lipstick", "airpods", "shampoo", "dry_flower", "wet_flower", "grapefruit", "avocado", "orange"]
    # obj = ["glass+beer", "glass+none", "glass+water", "eyeshadow", "toshiba", "clay", "lipstick", "airpods", "orange",
    #        "dry_flower", "wet_flower", "avocado", "bowl+soap", "bowl"]
    # obj = ["debit_card", "key", "green", "candle", "shampoo", "book", "bookmarker", "credit_card", "orange", "dry_flower", "wet_flower", "avocado", "bowl+soup", "bowl", "airpods", "glass+none", "glass+water", "eyeshadow", "clay", "lipstick"]
    # obj = ["debit card", "glass+water", "ceramic mean person", "magnet bookmark", "shampoo", "book", "glass", "credit card", "orange", "lipstick", "flower with enough water", "avocado", "bowl+soup", "bowl", "airpods", "clay", "eyeshadow", "flower lack of water", "candle", "mobile hard disk device"]
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
            peak = []
            line = f.readline()
            print("line: {}".format(line))
            iterms = line.strip('\n').split(',')
            # print("iterms[0]: {}, iterms[1]: {}".format(iterms[0], iterms[1]))

            load_diff = iterms[1:coilNums+1]
            load_diff = list(map(float, load_diff))
            trans_diff = iterms[coilNums+1:2*coilNums+1]
            trans_diff = list(map(float, trans_diff))
            trans_diff = [c / 50 for c in trans_diff]

            # extract the features: 23 types of~
            print("In file i={}, line j={}".format(i, j))
            # feature = feature_extraction8.feature_calculation(load_diff, trans_diff)  # i-which file; j-which line
            feature = feature_extraction8.feature_calculation(load_diff, trans_diff)  # i-which file; j-which line
            train_list.append(feature)  # one feature for each line
            print("i={}, obj[i]={}".format(i, obj[i]))
            label_list.append(obj[i])
        print("i've added the " + str(count) + "th object")

        f.close()

    # save the features for each sample and save them label in level.csv with same order.
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
        # print("len of current_line: {}".format(len(current_line.split(','))))
        # print("current_line: {}".format(current_line))
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


