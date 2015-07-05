# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:40:01 2015

@author: schaud7
"""

import pandas as pd
import numpy as np
import json
import LatLon
import logging
import math
import datetime

rows = 0
error_rows = 0

logging.basicConfig(filename='log_file.log',level=logging.DEBUG)

def getLastMile(x):
    global rows
    global error_rows
    rows += 1
    if((rows+1) % 100 == 0):
        logging.info(rows)
    try:
        l1 = LatLon.LatLon(LatLon.Latitude(x[-1][1]), LatLon.Longitude(x[-1][0]))
    except:
        error_rows += 1
    for i in xrange(len(x) - 1, -1, -1):
        l2 = LatLon.LatLon(LatLon.Latitude(x[i][1]), LatLon.Longitude(x[i][0]))
        dist = l1.distance(l2)/1.6
        if dist > 1.0:
            return(x[i:])
    return(x[1:])

# file loader function
def jsonLoader(x):
    loadLine = json.loads(x)
    return(loadLine)
    #loadLineSubset = getLastMile(loadLine)
    #return(loadLine[:1]+loadLineSubset)

# Load all the data files
test = pd.read_csv('./data/test.csv', converters={'POLYLINE': jsonLoader})
logging.info('rows: ' + str(rows))
logging.info('error rows: ' + str(error_rows))
sample = pd.read_csv('./data/sampleSubmission.csv')

# Convert polyline string to list of lists using json. You can also use: ast.literal_eval
#test['POLYLINE'] = test['POLYLINE'].apply(json.loads) 
#train['POLYLINE'] = train['POLYLINE'].apply(json.loads)

logging.info('loaded files....')

# Very crude way of generating some data here. I know lat is long and long is lat ;)
##### Crude method begins here
lat1 = []
long1 = []
#lat2 = []
#long2 = []
lat3 = []
long3 = []
lat_final = []
long_final = []
angle_cos = []
angle_sin = []
time_year = []
time_month = []
time_weekday = []
time_startsecs = []

def getAngle(lL1, lL2):
    l1 = LatLon.LatLon(LatLon.Latitude(lL1[1]), LatLon.Longitude(lL1[0]))
    l2 = LatLon.LatLon(LatLon.Latitude(lL2[1]), LatLon.Longitude(lL2[0]))
    return(l1.heading_initial(l2))
    
def timestampConv(stamp):
    conv_stamp = datetime.datetime.fromtimestamp(int(stamp))
    year = conv_stamp.year
    month = conv_stamp.month
    day = conv_stamp.day
    hour = conv_stamp.hour
    minute = conv_stamp.minute
    startsecs = conv_stamp.hour * 3600.0 + conv_stamp.minute * 60.0 + conv_stamp.second
    weekday = conv_stamp.weekday()
    return((year, month, day, hour, minute, startsecs, weekday))



lat1 = []
long1 = []
#lat2 = []
#long2 = []
lat3 = []
long3 = []
angle_cos = []
angle_sin = []
time_year = []
time_month = []
time_day = []
time_hour = []
time_min = []
time_weekday = []
time_startsecs = []

for i in range(len(test)):
    if((i+1) % 100 == 0):
        logging.info(i+1)
    try:
        lat1.append(test['POLYLINE'].values[i][0][1])
    except:
        lat1.append(-999)
    #try:
    #    lat2.append(test['POLYLINE'].values[i][1][1])
    #except:
    #    lat2.append(-999)
    try:
        lat3.append(test['POLYLINE'].values[i][-1][1])
    except:
        lat3.append(-999)

    try:
        long1.append(test['POLYLINE'].values[i][0][0])
    except:
        long1.append(-999)
    #try:
    #    long2.append(test['POLYLINE'].values[i][1][0])
    #except:
    #    long2.append(-999)
    try:
        long3.append(test['POLYLINE'].values[i][-1][0])
    except:
        long3.append(-999)
    try:
        temp_angle = getAngle(test['POLYLINE'].values[i][1], test['POLYLINE'].values[i][-1])
        angle_cos.append(math.cos(math.radians(temp_angle)))
        angle_sin.append(math.sin(math.radians(temp_angle)))
    except:
        angle_cos.append(-999)
        angle_sin.append(-999)
    # time conversion
    (year_temp, month_temp, day, hour, minute, startsecs_temp, weekday_temp) = timestampConv(test['TIMESTAMP'].values[i])
    time_year.append(year_temp)
    time_month.append(month_temp)
    time_day.append(day)
    time_hour.append(hour)
    time_min.append(minute)
    time_weekday.append(weekday_temp)
    time_startsecs.append(startsecs_temp)


logging.info('test complete...')
test['LAT1'] = lat1
#test['LAT2'] = lat2
test['LAT3'] = lat3
test['LONG1'] = long1
#test['LONG2'] = long2
test['LONG3'] = long3
test['ANGLE_COS'] = angle_cos
test['ANGLE_SIN'] = angle_sin
test['TIME_YEAR'] = time_year
test['TIME_MONTH'] = time_month
test['TIME_DAY'] = time_day
test['TIME_HOUR'] = time_hour
test['TIME_MIN'] = time_min
test['TIME_WEEKDAY'] = time_weekday
test['TIME_STARTSECS'] = time_startsecs
##### Crude method ends here

test['LEN'] = test['POLYLINE'].apply(lambda x: len(x))
test['TIME_ENDSECS'] = test['TIME_STARTSECS'] + test['LEN'].map(lambda x: 15.0*x)
test['END_HOUR'] = test['TIME_ENDSECS'].map(lambda x: x/3600)

bin_val = []
for i in range(25):
    bin_val.append(i)
    
out = pd.cut(test['END_HOUR'], bins = bin_val)
counts = pd.value_counts(out)
# counts is a Series
print(counts)

 #another way to get end hour:
test['TIMESTAMP_END'] =test['TIMESTAMP'].map(int) + test['LEN'].map(lambda x: 15.0*x)
test['END_HOUR_2'] = test['TIMESTAMP_END'].map(lambda x: datetime.datetime.utcfromtimestamp(x).hour)
test['END_DATE_2'] = test['TIMESTAMP_END'].map(lambda x: datetime.datetime.utcfromtimestamp(x).day)
test['END_WEEKDAY'] = test['TIMESTAMP_END'].map(lambda x: datetime.datetime.utcfromtimestamp(x).weekday())
pd.pivot_table(test, rows='END_DATE_2',cols='END_HOUR_2',values='TRIP_ID',aggfunc=len)
pd.pivot_table(test, rows='END_WEEKDAY',cols='END_HOUR_2',values='TRIP_ID',aggfunc=len)

# we should subset to take DAY_TYPE = A

# weekday break down
#TIME_WEEKDAY
#0               77
#1               77
#3               74
#4               62
#6               30


#TIME_MONTH
#8              74
#9              77
#10            139
#12             30

#NEW_COL2      3   4   6   7   8   9   13  14  22  23  24
#TIME_WEEKDAY                                            
#0            NaN NaN NaN NaN NaN NaN  77 NaN NaN NaN NaN
#1              1  76 NaN NaN NaN NaN NaN NaN NaN NaN NaN
#3            NaN NaN NaN NaN NaN NaN  22  52 NaN NaN NaN
#4            NaN NaN NaN NaN NaN NaN NaN NaN   1  22  39
#6            NaN NaN   2   2   1  25 NaN NaN NaN NaN NaN


#TIME_MONTH    8   9   10  12
#TIME_WEEKDAY                
#0            NaN NaN  77 NaN
#1            NaN  77 NaN NaN
#3             74 NaN NaN NaN
#4            NaN NaN  62 NaN
#6            NaN NaN NaN  30
#Maybe make 5 models with similar cutoffs




