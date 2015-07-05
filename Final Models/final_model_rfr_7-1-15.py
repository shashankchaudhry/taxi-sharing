# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:40:01 2015

@author: schaud7
"""

import pandas as pd
import numpy as np
import json
from sklearn import ensemble
import LatLon
import logging
import datetime
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from math import cos, sin, radians
import pdb

row_num = 0

logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)

#pdb.set_trace()
    
logging.info("started...")

def lineToLen(x):
    loadLine = json.loads(x)
    return(len(loadLine))
    
def timestampToDateTime(x):
    return(datetime.datetime.utcfromtimestamp(int(x)))

# Load the csv keeping only timestamp
timestamp = pd.read_csv('./data/train.csv', usecols = ['TIMESTAMP','POLYLINE'], converters={'POLYLINE': lineToLen,\
                         'TIMESTAMP': timestampToDateTime})
timestamp['YEAR'] = timestamp['TIMESTAMP'].apply(lambda x: x.year)
timestamp['MONTH'] = timestamp['TIMESTAMP'].apply(lambda x: x.month)
timestamp['STARTSECS'] = timestamp['TIMESTAMP'].apply(lambda x: x.hour * 3600.0 + x.minute * 60.0 + x.second)
timestamp['WEEKDAY'] = timestamp['TIMESTAMP'].apply(lambda x: x.weekday())
timestamp['ENDSECS'] = timestamp['STARTSECS'] + timestamp['POLYLINE'].map(lambda x: 15.0*x)

logging.info("time stamp done...")

SNAPSHOT_TIMES = [17.75, 8.5, 18.0, 4.0, 14.5, 17.25, 8.0, 9.0, 18.5, 3.5, 4.5, 14.0, 15.0, 19.0, \
                    2.5, 3.0, 5.0, 5.5, 15.5, 19.5, 13.5, 7.5, 9.5, 0.0, 1.0, 2.0, 6.0, 7.0, 10.5, 11.5, \
                    12.5, 16.0, 20.0, 21.0, 22.0, 23.0, 0.5, 1.5, 6.5, 10.0, 11.0, 12.0, 13.0, 16.5, 20.5 \
                    21.5, 22.5, 23.5]

for i,time in enumerate(SHAPSHOT_TIMES):
    SNAPSHOT_TIMES[i] = SNAPSHOT_TIMES[i] * 3600.0

def snapshotBetween(start_time, end_time):
    for time in SNAPSHOT_TIMES:
        if(time > start_time and time < end_time):
            return(time)
    return(None)

def getLastHalfMile(x):
    try:
        l1 = LatLon.LatLon(LatLon.Latitude(x[-1][1]), LatLon.Longitude(x[-1][0]))
    except:
        return([])
    for i in xrange(len(x) - 1, -1, -1):
        l2 = LatLon.LatLon(LatLon.Latitude(x[i][1]), LatLon.Longitude(x[i][0]))
        dist = l1.distance(l2)/1.6
        if dist > 0.5:
            return(x[i:])
    return(x[1:])
    
def getAnglePt(pt1, pt2):
    l1 = LatLon.LatLon(LatLon.Latitude(pt1[1]), LatLon.Longitude(pt1[0]))
    l2 = LatLon.LatLon(LatLon.Latitude(pt2[1]), LatLon.Longitude(pt2[0]))
    return(l1.heading_initial(l2))

def getAngle(line):
    if len(line) >= 6:
        mean_lat_1 = (line[0][1] + line[1][1] + line[2][1])/3.0
        mean_long_1 = (line[0][0] + line[1][0] + line[2][0])/3.0
        mean_lat_2 = (line[-1][1] + line[-2][1] + line[-3][1])/3.0
        mean_long_2 = (line[-1][0] + line[-2][0] + line[-3][0])/3.0
    elif len(line) >= 2:
        mean_lat_1 = line[0][1]
        mean_long_1 = line[0][0]
        mean_lat_2 = line[-1][1]
        mean_long_2 = line[-1][0]
    else:
        return(None)
    return(getAnglePt((mean_long_1,mean_lat_1),(mean_long_2,mean_lat_2)))

def getStartAngle(line):
    if len(line) >= 2:
        mean_lat_1 = line[0][1]
        mean_long_1 = line[0][0]
        mean_lat_2 = line[-1][1]
        mean_long_2 = line[-1][0]
    else:
        return(None)
    return(getAnglePt((mean_long_1,mean_lat_1),(mean_long_2,mean_lat_2)))

# Load all the data files
def trimLine(poly_line, start_time, end_time):
    cut_time = snapshotBetween(start_time, end_time)
    if(cut_time == None):
        return(([],0,-999,-999))
    new_time = start_time
    counter = 0
    while(new_time < cut_time):
        counter += 1
        new_time += 15
    try:
        tuple_val = (poly_line[counter+1],(counter+1))
        angle = getAngle(getLastHalfMile(poly_line[:counter+1]))
        if angle == None:
            return((tuple_val[0], tuple_val[1], -999, -999))
        return((tuple_val[0], tuple_val[1], cos(radians(angle)), sin(radians(angle))))
    except:
        return(([],0,-999,-999))
    
# file loader function
def jsonLoader(x):
    global row_num
    loadLine = json.loads(x)
    if(len(loadLine) > 0):
        start = loadLine[0]
        end = loadLine[-1]
        (cut,new_len,angle_cos,angle_sin) = trimLine(loadLine, timestamp['STARTSECS'].values[row_num], timestamp['ENDSECS'].values[row_num])
    else:
        start = []
        cut = []
        end = []
        new_len = 0
        angle_cos = -999
        angle_sin = -999
    row_num = row_num + 1
    if(row_num % 1000 == 0):
        logging.info(row_num)
    return((start,cut,end,new_len,angle_cos,angle_sin))

def simpleJsonLoader(x):
    loadLine = json.loads(x)
    angle = getAngle(getLastHalfMile(loadLine))
    start_angle = getStartAngle(loadLine)
    if start_angle == None:
        return((loadLine, -999, -999, -999, -999))
    elif angle == None:
        return((loadLine, -999, -999, cos(radians(start_angle)), sin(radians(start_angle))))
    return((loadLine, cos(radians(angle)), sin(radians(angle)), cos(radians(start_angle)), sin(radians(start_angle))))

train = pd.read_csv('./data/train.csv', converters={'POLYLINE': jsonLoader})
test = pd.read_csv('./data/test.csv', converters={'POLYLINE': simpleJsonLoader, 'TIMESTAMP': timestampToDateTime})
sample = pd.read_csv('./data/sampleSubmission.csv')

logging.info('loaded files....')

lat1 = []
long1 = []
lat3 = []
long3 = []
lat_final = []
long_final = []
new_length = []
cosine = []
sine = []
start_cos = []
start_sin = []

for i in range(len(train)):
    if((i+1) % 1000 == 0):
        logging.info(i+1)
    try:
        lat1.append(train['POLYLINE'].values[i][0][1])
    except:
        lat1.append(-999)
    try:
        lat3.append(train['POLYLINE'].values[i][1][1])
    except:
        lat3.append(-999)
    try:
        lat_final.append(train['POLYLINE'].values[i][2][1])
    except:
        lat_final.append(-999)

    try:
        long1.append(train['POLYLINE'].values[i][0][0])
    except:
        long1.append(-999)
    try:
        long3.append(train['POLYLINE'].values[i][1][0])
    except:
        long3.append(-999)
    try:
        long_final.append(train['POLYLINE'].values[i][2][0])
    except:
        long_final.append(-999)
    
    try:
        new_length.append(train['POLYLINE'].values[i][3])
    except:
        new_length.append(0)
    try:
        cosine.append(train['POLYLINE'].values[i][4])
    except:
        cosine.append(-999)
    try:
        sine.append(train['POLYLINE'].values[i][5])
    except:
        sine.append(-999)
    try:
        angle = getAnglePt((train['POLYLINE'].values[i][0][0], train['POLYLINE'].values[i][0][1]),\
                            (train['POLYLINE'].values[i][1][0], train['POLYLINE'].values[i][1][1]))
        start_cos.append(cos(radians(angle)))
        start_sin.append(sin(radians(angle)))
    except:
        start_cos.append(-999)
        start_sin.append(-999)

logging.info('train complete..')
train['LAT1'] = lat1
train['LAT3'] = lat3
train['LATF'] = lat_final
train['LONG1'] = long1
train['LONG3'] = long3
train['LONGF'] = long_final
train['NEW_LEN'] = new_length
train['COS'] = cosine
train['SIN'] = sine
train['START_COS'] = start_cos
train['START_SIN'] = start_sin
logging.info("train dim before drops: " + str(train.shape))

# append time stamp to train
train['TIME_YEAR'] = timestamp['YEAR']
train['TIME_MONTH'] = timestamp['MONTH']
train['TIME_WEEKDAY'] = timestamp['WEEKDAY']

# one-hot style call type
train['CALL_TYPE_A'] = train['CALL_TYPE'].apply(lambda x: 1 if x == 'A' else 0)
train['CALL_TYPE_B'] = train['CALL_TYPE'].apply(lambda x: 1 if x == 'B' else 0)
train['CALL_TYPE_C'] = train['CALL_TYPE'].apply(lambda x: 1 if x == 'C' else 0)

# drop some bad rows
train = train[train['LATF'] != -999]
train = train[train['LONGF'] != -999]
train = train[train['DAY_TYPE'] == 'A']
train = train[train['NEW_LEN'] > 0]
train = train[train['MISSING_DATA'] == False]
train = train[train['COS'] != -999]
train = train[train['SIN'] != -999]

logging.info("train dim after row drops: " + str(train.shape))

# drop trip_id, polyline, day_type and timestamp cols in train
train = train.drop(['TRIP_ID', 'DAY_TYPE', 'TIMESTAMP', 'POLYLINE', 'MISSING_DATA'], axis = 1)

logging.info("train dim after col drops: " + str(train.shape))

# make required cols for test
logging.info('test complete...')
test['LAT1'] = test['POLYLINE'].apply(lambda x: x[0][0][1])
test['LAT3'] = test['POLYLINE'].apply(lambda x: x[0][-1][1])
test['LONG1'] = test['POLYLINE'].apply(lambda x: x[0][0][0])
test['LONG3'] = test['POLYLINE'].apply(lambda x: x[0][-1][0])
test['TIME_YEAR'] = test['TIMESTAMP'].apply(lambda x: x.year)
test['TIME_MONTH'] = test['TIMESTAMP'].apply(lambda x: x.month)
test['TIME_WEEKDAY'] = test['TIMESTAMP'].apply(lambda x: x.weekday())
test['NEW_LEN'] = test['POLYLINE'].apply(lambda x: len(x[0]))
test['COS'] = test['POLYLINE'].apply(lambda x: x[1])
test['SIN'] = test['POLYLINE'].apply(lambda x: x[2])
test['START_COS'] = test['POLYLINE'].apply(lambda x: x[3])
test['START_SIN'] = test['POLYLINE'].apply(lambda x: x[4])
test['CALL_TYPE_A'] = test['CALL_TYPE'].apply(lambda x: 1 if x == 'A' else 0)
test['CALL_TYPE_B'] = test['CALL_TYPE'].apply(lambda x: 1 if x == 'B' else 0)
test['CALL_TYPE_C'] = test['CALL_TYPE'].apply(lambda x: 1 if x == 'C' else 0)

train = train.drop(['CALL_TYPE'], axis = 1)

# drop columns
test = test.drop(['CALL_TYPE', 'TRIP_ID', 'DAY_TYPE', 'TIMESTAMP', 'POLYLINE', 'MISSING_DATA'], axis = 1)

## reorder cols
train = train[['CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'LAT1', 'LAT3', 'LONG1', 'LONG3',\
       'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIME_MONTH',\
       'TIME_WEEKDAY', 'TIME_YEAR', 'COS', 'SIN', 'START_COS', 'START_SIN','NEW_LEN', 'LATF', 'LONGF']]
test = test[['CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'LAT1', 'LAT3', 'LONG1', 'LONG3',\
       'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIME_MONTH',\
       'TIME_WEEKDAY', 'TIME_YEAR', 'COS', 'SIN', 'START_COS', 'START_SIN', 'NEW_LEN']]

logging.info('modified lats and longs to keep....')

# factorize categorical columns in training set
for i in train.columns:
    if train[i].dtype == 'object':
        logging.info(i)
        train[i] = pd.factorize(train[i])[0]

# factorize categorical columns in test set
for i in test.columns:
    if test[i].dtype == 'object':
        logging.info(i)
        test[i] = pd.factorize(test[i])[0]

# fill all NaN values with -1
train = train.fillna(-1)
test = test.fillna(-1)

# Generate Labels and drop them from training set
label = np.array(train[['LATF', 'LONGF']])
train = train.drop(['LATF', 'LONGF'], axis = 1)
train = np.array(train)
test = np.array(test)

train = train.astype(float)
test = test.astype(float)

# custom scorer for haversine formula
def haversine_score(y, y_pred):
    counts = y.shape[0]
    net_dist = 0.
    for row_no in range(counts):
        l1 = LatLon.LatLon(LatLon.Latitude(y[row_no][0]), LatLon.Longitude(y[row_no][1]))
        l2 = LatLon.LatLon(LatLon.Latitude(y_pred[row_no][0]), LatLon.Longitude(y_pred[row_no][1]))
        net_dist += l1.distance(l2)
    return(net_dist / counts)

ss = cross_validation.ShuffleSplit(train.shape[0], n_iter=10, test_size=0.2, random_state=0)

logging.info('fitting to test data....')
results = []

msl = 5
logging.info("minimum samples leaf: " + str(msl))
clf = ensemble.RandomForestRegressor(n_estimators=800, n_jobs=1, min_samples_leaf=msl, verbose=1)
cross_val_prediction = cross_validation.cross_val_score(clf, train, label, cv=ss, scoring = make_scorer(haversine_score, greater_is_better=False), n_jobs=1)
mean_score = 0.
for i in cross_val_prediction:
   mean_score += cross_val_prediction[i]
mean_score = mean_score / len(cross_val_prediction)
logging.info('Mean CV score is : ' + str(mean_score))

clf.fit(train, label)
logging.info('Feature Importances: ' + str(clf.feature_importances_))
for item_no in range(test.shape[0]):
    test_case = test[item_no]
    results.append(clf.predict(test_case))

preds = np.asarray(results)
logging.info('writing file....')
# Write predictions to file
sample['LATITUDE'] = preds[:,0][:,0]
sample['LONGITUDE'] = preds[:,0][:,1]
sample.to_csv('./data/result_06-30-15_final_min_samples_leaf_5.csv', index = False)
