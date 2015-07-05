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
#import pdb

row_num = 0

#logging.basicConfig(filename='log_file.log',level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

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

# Load all the data files
def trimLine(poly_line, start_time, end_time):
    if(MONDAY_TIME > start_time and MONDAY_TIME < end_time):
        cut_time = MONDAY_TIME
    elif(TUESDAY_TIME > start_time and TUESDAY_TIME < end_time):
        cut_time = TUESDAY_TIME
    elif(THURSDAY_TIME > start_time and THURSDAY_TIME < end_time):
        cut_time = THURSDAY_TIME
    elif(FRIDAY_TIME > start_time and FRIDAY_TIME < end_time):
        cut_time = FRIDAY_TIME
    elif(SUNDAY_TIME > start_time and SUNDAY_TIME < end_time):
        cut_time = SUNDAY_TIME
    else:
        return(([],0))
    new_time = start_time
    counter = 0
    while(new_time < cut_time):
        counter += 1
        new_time += 15
    try:
        tuple_val = (poly_line[counter+1],(counter+1))
        return(tuple_val)
    except:
        return(([],0))
    
# file loader function
def jsonLoader(x):
    global row_num
    loadLine = json.loads(x)
    if(len(loadLine) > 0):
        start = loadLine[0]
        end = loadLine[-1]
        (cut,new_len) = trimLine(loadLine, timestamp['STARTSECS'].values[row_num], timestamp['ENDSECS'].values[row_num])
    else:
        start = []
        cut = []
        end = []
        new_len = 0
    row_num = row_num + 1
    if(row_num % 1000 == 0):
        logging.info(row_num)
    return((start,cut,end,new_len))

def simpleJsonLoader(x):
    loadLine = json.loads(x)
    return(loadLine)

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

logging.info('train complete..')
train['LAT1'] = lat1
train['LAT3'] = lat3
train['LATF'] = lat_final
train['LONG1'] = long1
train['LONG3'] = long3
train['LONGF'] = long_final
train['NEW_LEN'] = new_length

logging.info("train dim before drops: " + str(train.shape))

# append time stamp to train
train['TIME_MONTH'] = timestamp['MONTH']
train['TIME_WEEKDAY'] = timestamp['WEEKDAY']

# delete timestamp
# timestamp = None

# drop some bad rows
train = train[train['LATF'] != -999]
train = train[train['LONGF'] != -999]
train = train[train['DAY_TYPE'] == 'A']
train = train[train['NEW_LEN'] > 0]
train = train[train['MISSING_DATA'] == False]

logging.info("train dim after row drops: " + str(train.shape))

# drop trip_id, polyline, day_type and timestamp cols in train
train = train.drop(['ORIGIN_CALL','TRIP_ID', 'DAY_TYPE', 'TIMESTAMP', 'POLYLINE', 'MISSING_DATA'], axis = 1)

logging.info("train dim after col drops: " + str(train.shape))

# make required cols for test
logging.info('test complete...')
test['LAT1'] = test['POLYLINE'].apply(lambda x: x[0][1])
test['LAT3'] = test['POLYLINE'].apply(lambda x: x[-1][1])
test['LONG1'] = test['POLYLINE'].apply(lambda x: x[0][0])
test['LONG3'] = test['POLYLINE'].apply(lambda x: x[-1][0])
test['TIME_MONTH'] = test['TIMESTAMP'].apply(lambda x: x.month)
test['TIME_WEEKDAY'] = test['TIMESTAMP'].apply(lambda x: x.weekday())
test['NEW_LEN'] = test['POLYLINE'].apply(lambda x: len(x))

# drop columns
test = test.drop(['ORIGIN_CALL','TRIP_ID', 'DAY_TYPE', 'TIMESTAMP', 'POLYLINE', 'MISSING_DATA'], axis = 1)

## reorder cols
train = train[['CALL_TYPE', 'LAT1', 'LAT3', 'LONG1', 'LONG3',\
       'ORIGIN_STAND', 'TAXI_ID', 'TIME_MONTH',\
       'TIME_WEEKDAY', 'LATF', 'LONGF']]
test = test[['CALL_TYPE', 'LAT1', 'LAT3', 'LONG1', 'LONG3',\
       'ORIGIN_STAND', 'TAXI_ID', 'TIME_MONTH',\
       'TIME_WEEKDAY']]

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

# cross validation for selecting the best parameters
ss = cross_validation.ShuffleSplit(train.shape[0], n_iter=20, test_size=0.2, random_state=0)
param_grid_all={"n_estimators": [300, 500, 800]}

clf_rfr = ensemble.RandomForestRegressor(n_jobs=-1, verbose=1)
clf_etr = ensemble.ExtraTreesRegressor(n_jobs=-1, verbose=1)

gs_rfr = GridSearchCV(estimator=clf_rfr, cv=ss, param_grid=param_grid_all, scoring = make_scorer(haversine_score, greater_is_better=False), n_jobs=1)
gs_rfr.fit(train, label)
logging.info('grid search: random forest....')
logging.info('best score: ' + str(gs_rfr.best_score_))
logging.info('best params: ' + str(gs_rfr.best_params_))
best_score = gs_rfr.best_score_
best_params = gs_rfr.best_params_
model_no = 0
logging.info('all results: ' + str(gs_rfr.grid_scores_))

gs_etr = GridSearchCV(estimator=clf_etr, cv=ss, param_grid=param_grid_all, scoring = make_scorer(haversine_score, greater_is_better=False), n_jobs=1)
gs_etr.fit(train, label)
logging.info('grid search: extra tree regressor....')
logging.info('best score: ' + str(gs_etr.best_score_))
logging.info('best params: ' + str(gs_etr.best_params_))
if gs_etr.best_score_ < best_score:
   best_score = gs_etr.best_score_
   best_params = gs_etr.best_params_
   model_no = 1
logging.info('all results: ' + str(gs_etr.grid_scores_))

best_n_estimators = best_params['n_estimators']
best_max_features = best_params['max_features']

logging.info('building classifiers....')

if model_no == 0:
   clf = gs_rfr
elif model_no == 1:
   clf = gs_etr


logging.info('fitting to test data....')
results = []
clf = ensemble.RandomForestRegressor(n_estimators=800, n_jobs=-1, verbose=1)
clf.fit(train, label)
logging.info('Feature Importances: ' + str(clf.feature_importances_))
for item_no in range(test.shape[0]):
    test_case = test[item_no]
    results.append(clf.predict(test_case))
#preds = clf.predict(test)
preds = np.asarray(results)
logging.info('writing file....')
# Write predictions to file
sample['LATITUDE'] = preds[:,0][:,0]
sample['LONGITUDE'] = preds[:,0][:,1]
sample.to_csv('./data/result_06-21-15_1_after_grid_search.csv', index = False)
