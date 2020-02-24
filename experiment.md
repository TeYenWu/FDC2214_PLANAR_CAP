# Experiment LOG

### accuracy ? energy diverse?
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
data for each Object |10

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
data for each Object |10

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
data for each Object |10

- Poor performance. This may result in similarity among objects' energy.

### accuracy? amount of data?
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
SVM  | 0.06s|71.11
NB  | 0.017| 59.44
NN|0.02|48.89
LR|0.089|64.44
RF|1.14|**77.78**
DT|0.046|71.11

Object Num | 9
------------- | -------------|
data for each Object |20

- Although data amount is one time bigger than the last experiment, but top accuracy stays same.
- We may assume that data  amount is not so importance when it comes to accuracy in Random forest.

#### 20/02/23 21:41
| items        | energy   |
| --------   | -----:  |
| greenHead     | 0.27   |
| orange        |  1.12   |
| apple       |    0.8    |
|airpodsBOX | 0.3-0.8|

Model  |time(s)| accuracy(%)
------------- | -------------| -------------
SVM  | 0.01s|88.75
NB  | 0.014| 82.5
NN|0.0199|78.75
LR|0.0289|85
RF|1.20|**91.25**
DT|0.0179|85

Object Num | 4
------------- | -------------|
data for each Object |10

- top accuracy decrease tiny.
- Conclusion: More data != better performance.

### accuracy? load mode V.S transmission mode?
### Conclusion: Load > Tran > Load&Tran

#### 20/02/23 21:45 LOAD mode ONLY
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
SVM  | 0.05s|78.33
NB  | 0.017| 59.44
NN|0.0389|68.89
LR|0.1206|66.11
RF|1.16|**82.22**
DT|0.05|76.11

Object Num | 9
------------- | -------------|
data for each Object |20
- A litter better than load&transmission mode, whose top accuracy is 77.78. 82.22 > 77.78

#### 20/02/23 21:45 TRANSMISSION mode ONLY
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
SVM  | 0.064s|65
NB  | 0.029| 64.44
NN|0.036|49.44
LR|0.091|62.22
RF|1.2975|**79.44**
DT|0.04|71.11

Object Num | 9
------------- | -------------|
data for each Object |20
- A litter better than load&transmission mode, whose top accuracy is 77.78. 79.44 > 77.78
- Not as good as load-mode-only. 79.44 < 82.22


