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
from math import cos, sin, radians, floor, ceil
import pdb

row_num = 0

logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG)

MONDAY_TIME = 17.75 * 3600.0
TUESDAY_TIME = 8.5 * 3600.0
THURSDAY_TIME = 18.0 * 3600.0
FRIDAY_TIME = 4.0 * 3600.0
SUNDAY_TIME = 14.5 * 3600.0

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
    l1 = LatLon.LatLon(LatLon.Latitude(mean_lat_1), LatLon.Longitude(mean_long_1))
    l2 = LatLon.LatLon(LatLon.Latitude(mean_lat_2), LatLon.Longitude(mean_long_2))
    return(l1.heading_initial(l2))

def simpleJsonLoader(x):
    loadLine = json.loads(x)
    angle = getAngle(getLastHalfMile(loadLine))
    if angle == None:
        return((loadLine, -999, -999))
    return((loadLine, cos(radians(angle)), sin(radians(angle))))

test = pd.read_csv('./data/test.csv', converters={'POLYLINE': simpleJsonLoader, 'TIMESTAMP': timestampToDateTime})
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
test['CALL_TYPE_A'] = test['CALL_TYPE'].apply(lambda x: 1 if x == 'A' else 0)
test['CALL_TYPE_B'] = test['CALL_TYPE'].apply(lambda x: 1 if x == 'B' else 0)
test['CALL_TYPE_C'] = test['CALL_TYPE'].apply(lambda x: 1 if x == 'C' else 0)
# get final locations for test data:
final_locations = test['POLYLINE'].apply(lambda x: x[0][-1])
logging.info('see final locations: ' + str(final_locations))
# drop columns
test = test.drop(['CALL_TYPE', 'TRIP_ID', 'DAY_TYPE', 'TIMESTAMP', 'POLYLINE', 'MISSING_DATA'], axis = 1)

# get min and max test locations
min_lat = test['LAT3'].min() - 0.004
max_lat = test['LAT3'].max() + 0.004
min_long = test['LONG3'].min() - 0.005
max_long = test['LONG3'].max() + 0.005
max_lat_num = ceil((max_lat - min_lat) / 0.008)
max_long_num = ceil((max_long - min_long) / 0.01)

# a grid of 0.5 mi x 0.5 mi cells
# for every destination point, fill dict 
loc_dict = {}

def get_grid_loc(latitude, longitude):
    if((latitude < min_lat) or (latitude > max_lat)):
        return(None)
    elif((longitude < min_long) or (longitude > max_long)):
        return(None)
    else:
        lat_num = floor((latitude - min_lat) / 0.008)
        long_num = floor((longitude - min_long) / 0.01)
        grid_num = long_num + max_long_num * lat_num
        return(grid_num)

def set_grid_loc(latitude, longitude):
    global loc_dict
    grid_num = get_grid_loc(latitude, longitude)
    if grid_num in loc_dict:
        loc_dict[grid_num].append([latitude,longitude])
    else:
        loc_dict[grid_num] = [[latitude,longitude]]

for pair in final_locations.iteritems():
    set_grid_loc(pair[1][1], pair[1][0])

def getDistance(p1, p2):
    l1 = LatLon.LatLon(LatLon.Latitude(p1[1]), LatLon.Longitude(p1[0]))
    l2 = LatLon.LatLon(LatLon.Latitude(p2[1]), LatLon.Longitude(p2[0]))
    dist = l1.distance(l2)/1.6
    return(dist)

# new trimline function
# returns cut, new_len, cos, sin
def trimLine(poly_line, start_time, end_time):
    min_dist = float("inf")
    cut = None
    new_len = None
    possible_cuts = None
    length = 0
    for pair in poly_line:
        length += 1
        grid_loc = get_grid_loc(pair[1],pair[0])
        if length < 4:
            continue
        elif (min_dist != float("inf")):
            # at current location get min dist and compare to global min.
            # if it increased, return last vals, otherwise update and continue
            new_dist = float("inf")
            for test_pair in possible_cuts:
                dist = getDistance(pair, test_pair)
                if(dist < new_dist):
                    new_dist = dist
            if(new_dist < min_dist):
                min_dist = new_dist
                new_len = length
            else:
                break
        elif(grid_loc != None and grid_loc in loc_dict):
            possible_cuts = loc_dict[grid_loc]
            cut = [pair[0],pair[1]]
            new_len = length
            for test_pair in possible_cuts:
                dist = getDistance(pair, test_pair)
                if(dist < min_dist):
                    min_dist = dist
    if(min_dist == float("inf")):
        return(([],0,-999,-999))
    else:
        cut = poly_line[new_len - 1]
        try:
            angle = getAngle(getLastHalfMile(poly_line[:new_len]))
            if angle == None:
                return((cut, new_len, -999, -999))
            return((cut, new_len, cos(radians(angle)), sin(radians(angle))))
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

# now for training data
# for each polyline, check if 
train = pd.read_csv('./data/train.csv', converters={'POLYLINE': jsonLoader})
# save the effort to csv
train.to_csv('./data/train_after_cuts_06_23_15.csv', index=False)
logging.info('Saved cut results....')
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

logging.info("train dim before drops: " + str(train.shape))

# append time stamp to train
train['TIME_YEAR'] = timestamp['YEAR']
train['TIME_MONTH'] = timestamp['MONTH']
train['TIME_WEEKDAY'] = timestamp['WEEKDAY']

# one-hot style call type
train['CALL_TYPE_A'] = train['CALL_TYPE'].apply(lambda x: 1 if x == 'A' else 0)
train['CALL_TYPE_B'] = train['CALL_TYPE'].apply(lambda x: 1 if x == 'B' else 0)
train['CALL_TYPE_C'] = train['CALL_TYPE'].apply(lambda x: 1 if x == 'C' else 0)

# delete timestamp
# timestamp = None

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

train = train.drop(['CALL_TYPE'], axis = 1)

## reorder cols
train = train[['CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'LAT1', 'LAT3', 'LONG1', 'LONG3',\
       'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIME_MONTH',\
       'TIME_WEEKDAY', 'TIME_YEAR', 'COS', 'SIN', 'LATF', 'LONGF']]
test = test[['CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'LAT1', 'LAT3', 'LONG1', 'LONG3',\
       'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIME_MONTH',\
       'TIME_WEEKDAY', 'TIME_YEAR', 'COS', 'SIN']]

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
min_samples_leaf = [1,3,5]
logging.info('fitting to test data....')
results = []
for msl in min_samples_leaf:
    logging.info("minimum samples leaf: " + str(msl))
    clf = ensemble.RandomForestRegressor(n_estimators=800, n_jobs=1, min_samples_leaf=msl, verbose=1)
    cross_val_prediction = cross_validation.cross_val_score(clf, train, label, cv=ss, scoring = make_scorer(haversine_score, greater_is_better=False), n_jobs=1)
    mean_score = 0.
    for i in cross_val_prediction:
        mean_score += cross_val_prediction[i]
    mean_score = mean_score / len(cross_val_prediction)
    logging.info('Mean CV score is : ' + str(mean_score))
#clf.fit(train, label)
#logging.info('Feature Importances: ' + str(clf.feature_importances_))
#for item_no in range(test.shape[0]):
#    test_case = test[item_no]
#    results.append(clf.predict(test_case))
##preds = clf.predict(test)
#preds = np.asarray(results)
#logging.info('writing file....')
## Write predictions to file
#sample['LATITUDE'] = preds[:,0][:,0]
#sample['LONGITUDE'] = preds[:,0][:,1]
#sample.to_csv('./data/result_06-23-15_final_pos.csv', index = False)
