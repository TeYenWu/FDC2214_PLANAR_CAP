import numpy as np
import mahotas
import math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tsfresh
import time

#input data types:
#data base peak for each coils like[data1, data2, data3,... datan, base1, base2,... basen, peak1, peak2, peak3, ... peakn] n=? should be decided.
#

#parameters to be decided before run
r = 8
c = 8
channal = r * c
t= 20    # TODO: What's t? Gap between points
#rows = (r-1)*t
#cols=(c-1)*t
rows = r
cols = c
high=1
low=0
tempthreshold = 30  # origin:30

object_pixel=0
object_value=[]
mean_ten_section=[]
object_value=[0]*10


coil_array = np.zeros((r, c))
interp_array = np.zeros((rows, cols))
binary_array = np.zeros((rows, cols))
edge_array = np.zeros((rows, cols))
edge_numbers=0
peak = 90000000




def feature_calculation(data,base,imagename,imagename2):
    start_time=time.time()
    feature_set=[]
    ch=[]
    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers
    global value_section

    coil_array = np.zeros((r, c))
    interp_array = np.zeros((rows, cols))
    binary_array = np.zeros((rows, cols))
    edge_array = np.zeros((rows, cols))
    edge_numbers = 0
    object_pixel = 0
    object_value = []
    mean_ten_section = []

    # new coming features
    value_section=[0]*10
    # start_time = time.time()
    # ch denote the gray value(0-255) for each coils
    # ch here is float
    # print len(data)
    # print len(peak)
    '''
    make grey-scale map
    save in list ch, len(ch) = len(data)
    '''
    delta_list = list(map(lambda x: int(float(x[0]))-int(float(x[1])), zip(base, data)))
    max_delta = max(delta_list)
    # print("Max delta: {}".format(max_delta))
    for i in range(r*c):
        #print i
        if delta_list[i] <= 0:
            # ch.append(0.0)
            ch.append(0.0)
        else:
            t = (float(delta_list[i]))/max_delta
            t = t * 255
            ch.append(t)
            # print("delta {} / peak {} = ch {}".format(delta_data, peak[i], t))
    #print imagename,imagename2
    # print("ch")
    # print(ch)
    '''
    fill in np.array(coil_array[r][c]) with grey-scale map
    '''
    for y in range(r):  # row
        for x in range(c):  # column
            #print y,x
            coil_array[y][x] = ch[y*c + x]
            #print coil_array[y][x]
    # print("coil_array: {}".format(coil_array))

    bilinear_interpolation()    # produce interp_array[][], object_value[], value_section[]
    binary_image()  # from grey-scale to binary, only high / low. object_pixel
    edge_detection()    # get edge_array, values: high/low

    # A=np.array(interp_array)
    # im=Image.fromarray(A)
    # im = im.convert("L")
    # image1=str(imagename)+'_'+str(imagename2)+'.jpeg'
    # #print image1
    # im.save(image1)

    #now calculate the features:

    # 1. LBP 0-35
    # local binary pattern
    LBP_features = mahotas.features.lbp(interp_array, 2, 8)  # 2-D numpy ndarray; radius; points
    for i in LBP_features:
        feature_set.append(i)
    # print "extracted LBP feature with"+str(time.time()-start_time)
    #2. object_pixel 36
    feature_set.append(object_pixel)
    # print "extracted 2 feature with"+str(time.time()-start_time)
    #3. object_value 37-46
    mean_ten_section = means_for_each_part(object_value)
    for i in mean_ten_section:
        feature_set.append(i)
    #4. numbers of edge   :edge_numbers  47
    feature_set.append(edge_numbers)
    # print "extracted 4 feature with" + str(time.time() - start_time)
    #5.variance  48
    variance_pixel=np.var(object_value)
    feature_set.append(variance_pixel)
    # print "extracted 5 feature with" + str(time.time() - start_time)

    #6. ICP return the error of each training set value
    #these feature might be used as extra criterion

    #7. calculate the pixel numbers for each output section
    for i in value_section:
        feature_set.append(i)

    mi = 0.0
    xi_multiply_mi = 0.0
    yi_multiply_mi = 0.0
    # print "extracted 7 feature with" + str(time.time() - start_time)
    #8. average distance from each points to gravity point
    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x] == high:
                mi += interp_array[y][x]
                xi_multiply_mi = xi_multiply_mi+x*interp_array[y][x]
                yi_multiply_mi = yi_multiply_mi+y*interp_array[y][x]
    object_average = mi / object_pixel
    x_gravity=xi_multiply_mi/mi
    y_gravity=yi_multiply_mi/mi
    average_distance=0.0
    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x]==high:
                average_distance+=math.sqrt((x-x_gravity)*(x-x_gravity)+(y-y_gravity)*(y-y_gravity))/(1.0*object_pixel)
    feature_set.append(average_distance)
    # print "extracted 8 feature with" + str(time.time() - start_time)

    # 9. average distance form edge to gravity point
    average_edge_distance=0.0
    for y in range(rows):
        for x in range(cols):
            if edge_array[y][x]==high:
                average_edge_distance+=math.sqrt((x-x_gravity)*(x-x_gravity)+(y-y_gravity)*(y-y_gravity))/(1.0*edge_numbers)
    feature_set.append(average_edge_distance)

    xi=0.0
    yi=0.0
    # print "extracted 11 feature with" + str(time.time() - start_time)

    # 10. average distance from each points to geometry point
    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x]== high:
                xi=xi+x
                yi=yi+y
    x_geometry=xi/object_pixel
    y_geometry=yi/object_pixel
    average_distance_geo=0.0
    for y in range(rows):
        for x in range(cols):
            if binary_array[y][x]==high:
                average_distance_geo+=math.sqrt((x-x_geometry)*(x-x_geometry)+(y-y_geometry)*(y-y_geometry))/(1.0*object_pixel)
    feature_set.append(average_distance_geo)
    # print "extracted 12 feature with" + str(time.time() - start_time)

    # 11. average distance from edge points to geometry point
    average_distance_geo_edge=0.0
    for y in range(rows):
        for x in range(cols):
            if edge_array[y][x]==high:
                average_distance_geo_edge+=math.sqrt((x-x_geometry)*(x-x_geometry)+(y-y_geometry)*(y-y_geometry))/(1.0*object_pixel)
    feature_set.append(average_distance_geo_edge)
    # print "extracted 13 feature with" + str(time.time() - start_time)

    # 12.object average output
    feature_set.append(object_average)

    # 13. Hu moment
    moments = cv2.moments(interp_array)
    hu_moments = cv2.HuMoments(moments)
    for i in hu_moments:
        feature_set.append(i[0])

    # 14. max value of object:
    max_gray_value=np.max(interp_array)
    feature_set.append(max_gray_value)
    # print "extracted 11 feature with" + str(time.time() - start_time)
    # 15. local maximum
    extremum_nums=0
    for y in range(1,rows-1):
        for x in range(1,cols-1):
            if interp_array[y][x]==high :
                if interp_array[y][x]>=interp_array[y-1][x] and interp_array[y][x]>=interp_array[y][x-1] and interp_array[y][x]>=interp_array[y-1][x-1] and interp_array[y][x]>=interp_array[y+1][x] and interp_array[y][x]>=interp_array[y][x+1] and interp_array[y][x]>=interp_array[y-1][x+1] and interp_array[y][x]>=interp_array[y+1][x-1] and interp_array[y][x]>=interp_array[y+1][x+1]:
                    extremum_nums+=1
    feature_set.append(extremum_nums)
    # print "extracted 12 feature with" + str(time.time() - start_time)

    # 16. median value of object
    median_value=np.median(object_value)
    feature_set.append(median_value)
    # print "extracted 13 feature with" + str(time.time() - start_time)

    # 17. Quantiles
    quantiles =np.percentile(object_value, [25, 50, 75])
    for i in quantiles:
        feature_set.append(i)

    # 18
    feature_set.append(tsfresh.feature_extraction.feature_calculators.count_above_mean(object_value))

    # 19
    feature_set.append(tsfresh.feature_extraction.feature_calculators.count_below_mean(object_value))

    # 20
    # feature_set.append(tsfresh.feature_extraction.feature_calculators.sample_entropy(object_value))

    # 21
    feature_set.append(tsfresh.feature_extraction.feature_calculators.binned_entropy(object_value,2))

    # 22
    # feature_set.append(tsfresh.feature_extraction.feature_calculators.approximate_entropy(object_value,50,1))

    # 23
    feature_set.append(tsfresh.feature_extraction.feature_calculators.abs_energy(object_value))
    # print "extracted 15 feature with" + str(time.time() - start_time)

    return feature_set


def means_for_each_part(array_value):
    global r, c, channal, t, rows, cols, high, low, object_pixel, object_value, mean_ten_section
    global tempthreshold, coil_array, interp_array, binary_array, edge_array, edge_numbers
    array_value.sort()  # from small to big
    means = []
    step=len(array_value)/10
    for i in range(9):
        means.append(np.mean(array_value[int(i*step):int((i+1)*step)]))
    means.append(np.mean(array_value[int(9*step):]))
    return means


def bilinear_interpolation():
    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers
    global value_section
    # no problem
    t = 20
    '''
    for i in range(r):
        for j in range(c):
            x = j * t - 1
            y = i * t - 1
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            # print("Type of y:{}, value of y:{}, type of x:{}, value of x:{}".format(type(y), y, type(x), x))
            # print("x{} = j{} * t{} -1".format(x, j, t))
            # print("y{} = i{} * t{} -1".format(y, i, t))
            # print("i={}, j={}".format(i,j))
            interp_array[int(y)][int(x)] = coil_array[i][j]   # grey-scale-map
    '''

    for y in range(rows):
        for x in range(cols):
            interp_array[y][x] = coil_array[y][x]
            if interp_array[y][x] > tempthreshold:
                object_value.append(interp_array[y][x])

            if interp_array[y][x] == 255.0:  # new coming features
                value_section[9] += 1
            else:
                section = interp_array[y][x]/255.0
                value_section[int(section*10)] += 1
    # print("Interp_array: {}".format(interp_array))


def binary_image():

    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers

    for y in range(rows):
        for x in range(cols):
            if interp_array[y][x] < tempthreshold:
                binary_array[y][x] = low
            else:
                binary_array[y][x] = high
                object_pixel += 1
                #print object_pixel
    # print("object_pixel:{}".format(object_pixel))


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
    global r,c,channal,t,rows,cols,high,low,object_pixel,object_value,mean_ten_section
    global tempthreshold,coil_array,interp_array,binary_array,edge_array,edge_numbers

    maskvalue = binary_array[y - 1][x] + binary_array[y - 1][x + 1] + binary_array[y - 1][x - 1] + \
                binary_array[y][x + 1] + binary_array[y][x - 1] + binary_array[y + 1][x] + \
                binary_array[y + 1][x + 1] + binary_array[y + 1][x - 1]
    if maskvalue < 8 * high and maskvalue >= 1 * high:
        return high
    else:
        return low
