# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 21:08:37 2015

@author: schaud7
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
import json

bins = 5000
lat_min, lat_max = 41.0, 41.6
lon_min, lon_max = -8.75, -8.35

with ZipFile('data/train.csv.zip') as zf:
    
    data = pd.read_csv(zf.open('train.csv'),
                       chunksize=10000,
                       usecols=['POLYLINE'],
                       converters={'POLYLINE': lambda x: json.loads(x)})
    
    # process data in chunks to avoid using too much memory
    z = np.zeros((bins, bins))
    
    for i,chunk in enumerate(data):
        print(i)
        latlon = np.array([(path[-1][1], path[-1][0]) 
                           for path in chunk.POLYLINE
                           if len(path) > 0])

        z += np.histogram2d(*latlon.T, bins=bins, 
                            range=[[lat_min, lat_max],
                                   [lon_min, lon_max]])[0]
        
log_density = 5.0 * np.log(1+z)

plt.imshow(log_density[::-1,:], # flip vertically
           extent=[lon_min, lon_max, lat_min, lat_max])

# adding predicted final points
with open("data/result_06-26-15_more_times_min_samples_leaf_5 copy.csv",'r') as read_file:
    for i,line in enumerate(read_file):
        if i > 0:
            toks = line.split(',')
            plt.plot(float(toks[2]), float(toks[1]), 'rs', ms=0.02, fillstyle='full', linewidth = 0)
plt.savefig('heatmap.png',dpi=4000)