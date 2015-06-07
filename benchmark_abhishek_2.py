"""

Beating the Benchmark
ECML-PKDD Challenge 2015 @ Kaggle
__author__ : Abhishek Thakur

"""

import pandas as pd
import numpy as np
import json
from sklearn import ensemble

# file loader function
def jsonLoader(x):
    loadLine = json.loads(x)
    return(loadLine[:1]+loadLine[-2:])

# Load all the data files
train = pd.read_csv('./data/train.csv', converters={'POLYLINE': jsonLoader})
test = pd.read_csv('./data/test.csv', converters={'POLYLINE': jsonLoader})
sample = pd.read_csv('./data/sampleSubmission.csv')

# Convert polyline string to list of lists using json. You can also use: ast.literal_eval
#test['POLYLINE'] = test['POLYLINE'].apply(json.loads) 
#train['POLYLINE'] = train['POLYLINE'].apply(json.loads)

print('loaded files....')

# Very crude way of generating some data here. I know lat is long and long is lat ;)
##### Crude method begins here
lat1 = []
long1 = []
lat2 = []
long2 = []
lat_final = []
long_final = []

for i in range(len(train)):
    try:
        lat1.append(train['POLYLINE'].values[i][0][0])
    except:
        lat1.append(-999)
    try:
        lat2.append(train['POLYLINE'].values[i][-2][0])
    except:
        lat2.append(-999)
    try:
        lat_final.append(train['POLYLINE'].values[i][-1][0])
    except:
        lat_final.append(-999)

    try:
        long1.append(train['POLYLINE'].values[i][0][1])
    except:
        long1.append(-999)
    try:
        long2.append(train['POLYLINE'].values[i][-2][1])
    except:
        long2.append(-999)
    try:
        long_final.append(train['POLYLINE'].values[i][-1][1])
    except:
        long_final.append(-999)
    

train['LAT1'] = lat1
train['LAT2'] = lat2
train['LATF'] = lat_final
train['LONG1'] = long1
train['LONG2'] = long2
train['LONGF'] = long_final

lat1 = []
long1 = []
lat2 = []
long2 = []

for i in range(len(test)):
    try:
        lat1.append(test['POLYLINE'].values[i][0][0])
    except:
        lat1.append(-999)
    try:
        lat2.append(test['POLYLINE'].values[i][-1][0])
    except:
        lat2.append(-999)

    try:
        long1.append(test['POLYLINE'].values[i][0][1])
    except:
        long1.append(-999)
    try:
        long2.append(test['POLYLINE'].values[i][-1][1])
    except:
        long2.append(-999)

test['LAT1'] = lat1
test['LAT2'] = lat2
test['LONG1'] = long1
test['LONG2'] = long2
##### Crude method ends here

# drop some training data that doesnt have end-points
train = train[train['LATF'] != -999]
train = train[train['LONGF'] != -999]

# drop columns for benchmark model
train = train.drop(['TRIP_ID', 'TIMESTAMP', 'POLYLINE'], axis = 1)
test = test.drop(['TRIP_ID', 'TIMESTAMP', 'POLYLINE'], axis = 1)

print('modified lats and longs to keep....')

# factorize categorical columns in training set
for i in train.columns:
    if train[i].dtype == 'object':
        print i
        train[i] = pd.factorize(train[i])[0]

# factorize categorical columns in test set
for i in test.columns:
    if test[i].dtype == 'object':
        print i
        test[i] = pd.factorize(test[i])[0]

# fill all NaN values with -1
train = train.fillna(-1)
test = test.fillna(-1)

# Generate Labels and drop them from training set
labels = np.array(train[['LATF', 'LONGF']])
train = train.drop(['LATF', 'LONGF'], axis = 1)
train = np.array(train)
test = np.array(test)

train = train.astype(float)
test = test.astype(float)

print('building classifier....')
# Initialize the famous Random Forest Regressor from scikit-learn
clf = ensemble.RandomForestRegressor(n_jobs=-1, n_estimators=100)
clf.fit(train, labels)
print('fitting to test data....')
preds = clf.predict(test)

print('writing file....')
# Write predictions to file
sample['LATITUDE'] = preds[:,1]
sample['LONGITUDE'] = preds[:,0]
sample.to_csv('./data/benchmark_2.csv', index = False)