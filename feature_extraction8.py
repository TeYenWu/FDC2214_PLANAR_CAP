import numpy as np
import mahotas
import math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tsfresh
import time

r = 12
c = 12
channal = r * c
t = 20    # TODO: What's t? Gap between points

rows = r * 5
cols = c * 5
high = 255
low = 0
tempthreshold = 125  # 255 * 0.5    # TODO

object_pixel = 0
object_value = []
mean_ten_section = []

coil_array = np.zeros((r, c))
interp_array = np.zeros((rows, cols))
binary_array = np.zeros((rows, cols))
edge_array = np.zeros((rows, cols))
edge_numbers = 0

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 1
params.maxArea = 100
params.filterByColor = True
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)

def feature_calculation(load_diff, trans_diff):
    load_diff = list(map(float, load_diff))
    start_time = time.time()
    feature_set = []
    ch = []
    global r, c, channal, t, rows, cols, high, low, object_pixel, object_value, mean_ten_section, load_interp_array
    global tempthreshold, coil_array, interp_array, binary_array, edge_array, edge_numbers
    global value_section
    coil_array = np.zeros((r, c))
    interp_array = np.zeros((r, c))
    load_interp_array = np.zeros((r, c))
    binary_array = np.zeros((rows, cols))
    edge_array = np.zeros((rows, cols))
    edge_numbers = 0
    object_pixel = 0
    object_value = []
    mean_ten_section = []
    # new coming features
    value_section = [0]*10
    '''
    make grey-scale map for transmission mode
    save in list ch, len(ch) = len(data)
    '''
    # print("trans_diff[i]:{}".format(trans_diff))
    # print("length of trans_diff[i]:{}".format(len(trans_diff)))
    for i in range(r*c):
        # print("i={}".format(i))
        if float(trans_diff[i]) <= 0:
            ch.append(0.0)
        else:
            t = trans_diff[i]
            ch.append(t)
    '''
    fill in np.array(coil_array[r][c]) with grey-scale map
    '''
    for y in range(r):  # row
        for x in range(c):  # column
            # print y,x
            coil_array[y][x] = ch[y*c + x]

    for y in range(r):  # row
        for x in range(c):  # column
            load_interp_array[y][x] = load_diff[y*c + x]

    bilinear_interpolation(load_diff)    # produce interp_array[][], object_value[] (del-value_section[])
    # interp_array = (interp_array * 25.5).astype('uint8')
    # interp_array = np.array(interp_array*25.5, dtype='uint8')
    interp_array = cv2.resize(interp_array, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC )
    # load_interp_array = (load_interp_array * 25.5).astype('uint8')
    # load_interp_array = np.array(load_interp_array * 25.5, dtype='uint8')
    load_interp_array = cv2.resize(load_interp_array, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC)
    # print(load_interp_array)
    # cv2.imshow("test", load_interp_array)
    # cv2.waitKey(0)
    binary_image()  # from grey-scale to binary, only high / low. get binary_array[y][x], object_pixel
    edge_detection()    # get edge_array, only_edge_array[y][x] . values: high/low


    # now calculate the features:

    # 1. LBP 0-35
    # local binary pattern
    # update 02/24: filter noise position out of feature set
    LBP_features = mahotas.features.lbp(interp_array, 3, 8)  # 2-D numpy ndarray; radius; points
    for i in LBP_features:
        feature_set.append(i)
    # print("------1. LBP features: {}".format(feature_set))
    # print "extracted LBP feature with"+str(time.time()-start_time)

    # 2. object_pixel 36
    feature_set.append(object_pixel)
    # print("------2. object_pixel: {}".format(object_pixel))
    # print "extracted 2 feature with"+str(time.time()-start_time)

    # 3. object_value 37-46
    mean_ten_section = means_for_each_part(object_value)
    for i in mean_ten_section:
        feature_set.append(i)
    # print("------3. energy mean of 10 sections:{}".format(mean_ten_section))

    # 4. numbers of edge  [47]
    feature_set.append(edge_numbers)
    # print("------4.edge_numbers: {}".format(edge_numbers))
    # print "extracted 4 feature with" + str(time.time() - start_time)

    # 5.variance  [48]
    variance_pixel = np.var(object_value)
    feature_set.append(variance_pixel)
    # print("------5: variance: {}".format(variance_pixel))
    # print "extracted 5 feature with" + str(time.time() - start_time)

    # 6. ICP return the error of each training set value
    # these feature might be used as extra criterion

    # 7. calculate the pixel numbers for each output section
    value_section = count_for_each_part(object_value)
    for i in value_section:
        feature_set.append(i)
    # print("------7: counts of energy sections: {}".format(value_section))
    # print "extracted 7 feature with" + str(time.time() - start_time)

    # 8. average distance from each points to energy-center point
    sum_energy = 0.0
    trans_sum_energy = 0.0
    x_sum_energy = 0.0
    y_sum_energy = 0.0

    # print("load_diff:{}".format(load_diff))
    # print("rows:{}, cols:{}".format(rows,cols))
    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x] == high:
                sum_energy += load_interp_array[y][x]
                x_sum_energy += x * load_interp_array[y][x]
                y_sum_energy += y * load_interp_array[y][x]
                trans_sum_energy += interp_array[y][x]

    object_average = sum_energy / object_pixel
    trans_object_avr = trans_sum_energy / object_pixel
    # print("x_sum_energy:{} / sum_energy:{}".format(x_sum_energy, sum_energy))
    if sum_energy == 0:
        x_gravity = 0
        y_gravity = 0
    else:
        x_gravity = x_sum_energy / sum_energy
        y_gravity = y_sum_energy / sum_energy
    average_distance = 0.0

    feature_set.append(x_gravity)
    feature_set.append(y_gravity)
    # print("x_gravity:{}, y_gravity:{}".format(x_gravity, y_gravity))

    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x] == high:
                average_distance += math.sqrt((x-x_gravity)**2 + (y-y_gravity)**2) / (1.0*object_pixel)
    feature_set.append(average_distance)
    # print("------8: average of OBJECT pixels to ENERGY center: {}".format(average_distance))
    # print "extracted 8 feature with" + str(time.time() - start_time)

    # 9. average distance form edge to gravity point
    average_edge_distance = 0.0
    for y in range(rows):
        for x in range(cols):
            if edge_array[y][x] == high:
                average_edge_distance += math.sqrt((x-x_gravity)**2+(y-y_gravity)**2)/(1.0*edge_numbers)
    feature_set.append(average_edge_distance)
    # print("------9: average of EDGE pixels to ENERGY center: {}".format(average_edge_distance))     # todo: why always zero
    # print "extracted 11 feature with" + str(time.time() - start_time)

    # 10. average distance from each points to geometry point
    xi = 0.0
    yi = 0.0
    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x] == high:
                xi += x
                yi += y
    x_geometry = xi/object_pixel
    y_geometry = yi/object_pixel
    # print("x_geometry:{},y_geometry:{}".format(x_geometry, y_geometry))
    feature_set.append(x_geometry)
    feature_set.append(y_geometry)
    # print("x_geometry:{}, y_geometry:{}".format(x_geometry, y_geometry))
    average_distance_geo = 0.0
    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x] == high:
                average_distance_geo += math.sqrt((x-x_geometry)**2+(y-y_geometry)**2)/(1.0*object_pixel)
    feature_set.append(average_distance_geo)

    # Setup SimpleBlobDetector parameters.
    cvimage = (binary_array).astype('uint8')
    keypoints = detector.detect(cvimage)

    keypoints_sizes = [keypoint.size for keypoint in keypoints]
    feature_set.append(len(keypoints))
    feature_set.append(sum(keypoints_sizes)/len(keypoints) if len(keypoints) != 0 else 0)
    # print("------ num of keypoints: {}".format(len(keypoints)))
    # print("------10: average of OBJECT pixels to GEOMETRIC center: {}".format(average_distance_geo))
    # print "extracted 12 feature with" + str(time.time() - start_time)

    # 11. average distance from edge points to geometry point
    average_distance_geo_edge = 0.0
    for y in range(rows):
        for x in range(cols):
            if edge_array[y][x] == high:
                average_distance_geo_edge += math.sqrt((x-x_geometry)**2+(y-y_geometry)**2)/(1.0*object_pixel)
    feature_set.append(average_distance_geo_edge)
    # print("------11: average of EDGE pixels to GEOMETRIC center: {}".format(average_distance_geo))
    # print "extracted 13 feature with" + str(time.time() - start_time)

    # 12.object average output
    feature_set.append(object_average)  # average of energy
    feature_set.append(trans_object_avr)    # average energy for transmission mode
    # print("------12: mean of energy LOAD MODE{}".format(object_average))
    # print("------12: mean of energy TRANS MODE{}".format(trans_object_avr))

    # 13. Hu moment
    moments = cv2.moments(interp_array)  # interp_array:transmission data
    hu_moments = cv2.HuMoments(moments)
    for i in hu_moments:
        feature_set.append(i[0])
    # print("------13: Hu moment:", end=" ")
    # for i in hu_moments:
    #     print(i[0], end=',')
    # print("\n")

    # 14. max value of object:s
    max_gray_value = max(load_diff)
    feature_set.append(max_gray_value)
    # print("------14: max energy:{}".format(max_gray_value))
    # print "extracted 11 feature with" + str(time.time() - start_time)

    # 15. local maximum
    extremum_nums = 0
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            if binary_array[y][x] == high:
                if interp_array[y][x] >= interp_array[y-1][x] and interp_array[y][x] >= interp_array[y][x-1] and interp_array[y][x] >= interp_array[y-1][x-1] and interp_array[y][x] >= interp_array[y+1][x] and interp_array[y][x] >= interp_array[y][x+1] and interp_array[y][x] >= interp_array[y-1][x+1] and interp_array[y][x]>=interp_array[y+1][x-1] and interp_array[y][x]>=interp_array[y+1][x+1]:
                    extremum_nums += 1
    feature_set.append(extremum_nums)
    # print("------15:(del)local max energy counts:{}".format(extremum_nums))     # TODO:why always zero?
    # print "extracted 12 feature with" + str(time.time() - start_time)

    # 16. median value of object
    median_value = np.median(object_value)
    feature_set.append(median_value)
    # print("------16: median object energy:{}".format(median_value))
    # print "extracted 13 feature with" + str(time.time() - start_time)

    # 17. Quantiles
    quantiles = np.percentile(object_value, [25, 50, 75])
    for i in quantiles:
        feature_set.append(i)
    # print("------17: quantiles:{}".format(quantiles))

    # 18
    c_a_m = tsfresh.feature_extraction.feature_calculators.count_above_mean(object_value)
    feature_set.append(c_a_m)
    # print("------18: count_above_mean:{}".format(c_a_m))

    # 19
    c_b_m = tsfresh.feature_extraction.feature_calculators.count_below_mean(object_value)
    feature_set.append(c_b_m)
    # print("------19: count_below_mean:{}".format(c_b_m))

    # 20
    # feature_set.append(tsfresh.feature_extraction.feature_calculators.sample_entropy(object_value))

    # 21
    binned_entropy = tsfresh.feature_extraction.feature_calculators.binned_entropy(object_value, 2)
    feature_set.append(binned_entropy)  # todo: max number of bins:2?
    # print("------21: binned_entropy:{}".format(binned_entropy))

    # 22
    # feature_set.append(tsfresh.feature_extraction.feature_calculators.approximate_entropy(object_value,50,1))

    # 23 absolute energy of OBJECT pixel
    #  sum over the squared values
    abs_energy = tsfresh.feature_extraction.feature_calculators.abs_energy(object_value)
    feature_set.append(abs_energy)
    # print("------23: abs_energy:{}".format(abs_energy))
    # print "extracted 15 feature with" + str(time.time() - start_time)
    return feature_set


def means_for_each_part(array_value):
    # print("object_value: {}".format(array_value))
    global r, c, channal, t, rows, cols, high, low, object_pixel, object_value, mean_ten_section
    global tempthreshold, coil_array, interp_array, binary_array, edge_array, edge_numbers
    array_value.sort()  # from small to big
    means = []
    step = len(array_value)/10  # todo: decrease 10 to ? so that no nan would exsist
    for i in range(9):
        means.append(np.mean(array_value[int(i*step):int((i+1)*step)]))
        # print("Values: {}".format(array_value[int(i*step):int((i+1)*step)]))
    means.append(np.mean(array_value[int(9*step):]))
    return means


def count_for_each_part(array_value):
    # print("line 300 object_value: {}".format(array_value))
    # print("object_value: {}".format(array_value))
    global r, c, channal, t, rows, cols, high, low, object_pixel, object_value, mean_ten_section
    global tempthreshold, coil_array, interp_array, binary_array, edge_array, edge_numbers
    array_value.sort()  # from small to big
    counts = []
    step = len(array_value) / 10  # todo: decrease 10 to ? so that no nan would exsist
    for i in range(9):
        counts.append(len(array_value[int(i * step):int((i + 1) * step)]))
        # print("counts: {}".format(array_value[int(i*step):int((i+1)*step)]))
    counts.append(len(array_value[int(9 * step):]))
    return counts


def bilinear_interpolation(load_diff):
    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers
    global value_section
    # load_diff = list(map(float, load_diff))
    for y in range(r):
        for x in range(c):
            interp_array[y][x] = coil_array[y][x]
            # if interp_array[y][x] > 0:  # position proof
            #     object_value.append(load_diff[y * r + x])  # energy of the point

    # print("object_value: {}".format(object_value))

            # if interp_array[y][x] == 255.0:  # new coming features
            #     value_section[9] += 1  # TODO:what's value_section[]
            # else:
            #     section = interp_array[y][x]/255.0  # TODO:what's section
            #     # value_section[int(section*10)] += 1
    # print("Interp_array: {}".format(interp_array))


def binary_image():

    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section, load_interp_array
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers

    threshold = 0.25 * interp_array.max()
    for y in range(rows):
        for x in range(cols):
            #if interp_array[y][x] < tempthreshold:
            if interp_array[y][x] > threshold:
                binary_array[y][x] = high
                object_pixel += 1
                object_value.append(load_interp_array[y][x])  # energy of the point
            else:
                binary_array[y][x] = low


                #print object_pixel
    # print("object_value:{}".format(object_value))


def edge_detection():
    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers
    edge_array = np.zeros((rows, cols))
    only_edge_array = [[], []]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if binary_array[y][x] == high and mask(y, x) == high:
                edge_array[y][x] == high
                only_edge_array[0].append(y)
                only_edge_array[1].append(x)
                edge_numbers += 1
            else:
                edge_array[y][x] == low


def mask(y,x):
    """Filter noise points
    """

    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers

    maskvalue = binary_array[y - 1][x] + binary_array[y - 1][x + 1] + binary_array[y - 1][x - 1] + \
                binary_array[y][x + 1] + binary_array[y][x - 1] + binary_array[y + 1][x] + \
                binary_array[y + 1][x + 1] + binary_array[y + 1][x - 1]
    if maskvalue < 8 * high and maskvalue >= 1 * high:
        return high
    else:
        return low
