import json
#import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/schaud7/Documents/personal_git_taxi_demand/data/test.csv', 
                 converters={'POLYLINE': lambda x: json.loads(x)})

#lat_low = 41.123232
#lat_hgh = 41.237424
#lon_low = -8.687466
#lon_hgh = -8.553186
lat_low = 41.0
lat_hgh = 41.6
lon_low = -8.75
lon_hgh = -8.35

imp_pts = ['T7','T10','T11','T14','T71','T72','T77','T88','T96','T98','T116','T127','T128','T146','T152','T158',\
'T172','T182','T208','T229','T230','T232','T235','T259','T261','T276','T279','T295','T299','T303','T307','T315','T321']

df = df[df['TRIP_ID'].isin(imp_pts)]
fig = plt.figure()
ax = fig.add_subplot(111)
cm=plt.get_cmap('prism')
color=iter(cm(np.linspace(0,1,327)))
plt.ylim((lat_low,lat_hgh))
plt.xlim((lon_low,lon_hgh))
for i,p in enumerate(df['POLYLINE']):
    lats = []
    longs = []
    c=next(color)
    if len(p)>0:
        for pair in p:
            lats.append(pair[1])
            longs.append(pair[0])
        #plt.figure()
        #plt.scatter(longs,lats)
        # stop code here
        #temp = input('press enter: ')
        plt.plot(longs, lats, '.-', c=c, ms=0.5, lw=0.1)
        plt.plot(longs[-1],lats[-1],'ro', ms=0.8,mfc=None)
        ax.annotate('(%s)' % (imp_pts[i]), xy=(longs[-1],lats[-1]),size= 1)
plt.savefig('test_paths.png',dpi=4000)
plt.show()

