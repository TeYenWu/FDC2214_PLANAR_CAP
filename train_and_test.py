# -*- coding: utf-8 -*-
# 用于将收集的数据做成模型，然后进行预测。

import feature_extraction8
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, ShuffleSplit
#from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import random
from sklearn.externals import joblib
import os
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# 各种分类函数

TYPE_NUM = 7   # todo: changed me to count(object)

# SVM Classifier


def svm_classifier():
    from sklearn.svm import SVC
    model = SVC(kernel='linear', C=1)
    return model


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier():
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    return model


# KNN Classifier
def knn_classifier():
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    return model


# NearestNeighbors
def nearest_neighbors():
    from sklearn import neighbors
    model = neighbors.KNeighborsClassifier(3, 'distance')
    return model


# Random Forest Classifier
def random_forest_classifier():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_features="auto")
    return model


# Decision Tree Classifier
def decision_tree_classifier():
    from sklearn import tree
    model = tree.DecisionTreeClassifier(criterion='entropy', max_features=0.4, max_depth=15, splitter='best')
    # model = tree.DecisionTreeClassifier(criterion = 'entropy',max_features=0.1, max_depth=10, splitter='best')
    return model


## 将数据保存为数组形式
def random_shuffle_data(data, labels, seedNum):
    random.seed(seedNum)
    ind = range(0, len(data), 1)
    random.shuffle(ind)
    shuffleData = []
    shuffleLabels = []
    for i in ind:
        shuffleData.append(data[i])
        shuffleLabels.append(labels[i])
    return shuffleData, shuffleLabels


def print_confusion_matrix(y_true, y_pred):
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print("labels\t,")
    for i in range(len(labels)):
        print(labels[i] + "\t",)

    for i in range(len(conf_mat)):
        print(i, "\t",)
        for j in range(len(conf_mat[i])):
            print(conf_mat[i][j], '\t',)



def plot_confusion_matrix(y_true, y_pred, labels, saveFileName, showFlag):
    # print("labels: {}".format(labels))
    # 用于将预测结果表示出来。
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 12), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0  # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=7, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                # 这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=6, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=6, va='center', ha='center')
    if (intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of actual classes', fontsize=20)
    plt.xlabel('Index of predicted classes', fontsize=20)
    plt.savefig(saveFileName, dpi=300)
    if showFlag:
        plt.show()


if __name__ == '__main__':
    print('begin myclassifier')

    objectNums = TYPE_NUM
    trainDataNums = 20
    coilNums = 144

    # Make train_list(float list) from file:features.csv
    train_list = []
    label_list = []

    cwd = os.getcwd()  # current working dictionary
    feature_file = cwd + '\\coil\\data\\' + 'features.csv'
    f_feature = open(feature_file, 'r')
    lines = f_feature.readlines()
    for i in lines:
        iterm_feature = i.strip('\n').split(',')
        for k in range(len(iterm_feature)):
            iterm_feature[k] = float(iterm_feature[k])
        train_list.append(iterm_feature)
    f_feature.close()
    # Make label_list(int list) from file:label.csv
    label_file = cwd + '\\coil\\data\\' + 'label.csv'
    f_label = open(label_file, 'r')
    line = f_label.readline()
    iterm_label = line.strip('\n').split(',')
    # iterm_label = line.split(',')
    for k in range(len(iterm_label)):
        # iterm_label[k] = int(iterm_label[k])
        label_list.append(iterm_label[k])

    # 将数据改成ndarray格式
    x = np.array(train_list, dtype=np.float64)
    y = np.array(label_list)
    ## 数据预处理：归一化处理
    minMaxScale = preprocessing.MinMaxScaler()
    x = minMaxScale.fit_transform(x)
    joblib.dump(minMaxScale, "scalar.save")
    # test_classifiers = ['SVM', 'NB', 'NN', 'LR', 'RF', 'DT']  # 'SVM','NB','KNN','NN', 'LR', 'RF', 'DT'
    test_classifiers = ['RF']
    classifiers = {
        'SVM': svm_classifier,
        'NB': naive_bayes_classifier,
        'KNN': knn_classifier,
        'NN': nearest_neighbors,
        'LR': logistic_regression_classifier,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier
    }

    ## 使用不同分类方法对实验数据进行仿真
    misNumPerClass = {}  # 存储不同类别的各自错误数
    classNum = TYPE_NUM  # 数据集的类别个数。需要根据
    for i in range(1, classNum + 1, 1):
        misNumPerClass[i] = 0
    # KFoldNum存储交叉验证的次数。
    KFoldNum = 2

    # train and store a Random Forest classifier.
    RF = RandomForestClassifier(n_estimators=100, max_features="auto")
    from sklearn import tree
    x = np.nan_to_num(x)
    RF.fit(x, y)
    print(os.getcwd())
    joblib.dump(RF, 'RF2.model')

    for classifier in test_classifiers:  # calculate the confusion matrix and show the accuracy for different classifier.
        print('******************* %s ********************' % classifier)
        individual_accuracy = [0] * TYPE_NUM
        start_time = time.time()
        model = classifiers[classifier]()
        predict = cross_val_predict(model, x, y, cv=KFoldNum)

        print('cross_val_predict took %fs!' % (time.time() - start_time))
        accuracy = metrics.accuracy_score(y, predict)
        # print("predict:\n{}".format(predict))
        for i in range(len(predict)):
            if i%20 == 0:
                print("\n")
            print(predict[i], end=',')
            if predict[i] == y[i]:
                # print("i//20={}".format(i//20))
                individual_accuracy[i//20] += 1

        # print("\ny:\n{}".format(y))
        # print_confusion_matrix(y,predict)

        # save a confusion matrix as .png
        cm = confusion_matrix(y, predict)
        figFileName = classifier + '_cm.png'

        showFig = 0
        plot_confusion_matrix(y, predict, list(set(label_list)), figFileName, showFig)
        # print training accuracy.
        print('\naccuracy: %.2f%%' % (100 * accuracy))

        print("individual_accuracy[]:{}".format(individual_accuracy))
        individual_accuracy = [c / 20 for c in individual_accuracy]
        print("individual_accuracy[]:{}".format(individual_accuracy))
        ac_std = np.std(individual_accuracy, ddof=1)
        print("accuracy SD={}".format(ac_std))
