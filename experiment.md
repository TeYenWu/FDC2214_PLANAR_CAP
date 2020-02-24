# Experiment LOG

#### 20/02/23 20：58
| items        | energy   |
| --------   | -----:  |
| greenHead     | 0.27   |
| orange        |  1.12   |
| apple       |    0.8    |
|airpodsBOX | 0.3-0.8|
|airpods| 0.044|
|halfwater | 0.3|
|fullwater | 0.36|
|tinyTomato|0.28-0.54|
|tinyGrape | 0.36|

Model  |time(s)| accuracy(%)
------------- | -------------|------------
SVM  | 0.03s|56.67
NB  | 0.017| 57.78
NN|0.02|40
LR|0.052|58.89
RF|1.14|**77.78**
DT|0.031|72.22

Object Num | 9
------------- | -------------|
data for each Object |20

- Some of these 9 objects looks similar in energy, which maybe the reason for the poor performance.

#### 20/02/23 21:05
| items        | energy   |
| --------   | -----:  |
| greenHead     | 0.27   |
| orange        |  1.12   |
| apple       |    0.8    |
|airpodsBOX | 0.3-0.8|

Model  |time(s)| accuracy(%)
------------- | -------------| -------------
SVM  | 0.01s|82.5
NB  | 0.014| 72.5
NN|0.0169|72.5
LR|0.0189|77.5
RF|0.8297|**95**
DT|0.0179|87.5

Object Num | 4
------------- | -------------|
data for each Object |20

- Performance greatly improved, this may result in greater diverse among objects.

#### 20/02/23 21:17
| items        | energy   |
| --------   | -----:  |
|halfwater | 0.3|
|fullwater | 0.36|
|tinyTomato|0.28-0.54|
|tinyGrape | 0.36|

Model  |time(s)| accuracy(%)
------------- | -------------| -------------
SVM  | 0.01s|62.5
NB  | 0.014| 57.5
NN|0.0169|50
LR|0.0189|65
RF|0.8297|**70**
DT|0.0179|60

Object Num | 4
------------- | -------------|
data for each Object |20

- Poor performance. This may result in similarity among objects' energy.
