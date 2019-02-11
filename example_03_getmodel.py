import sys, os
import numpy as NP
import pandas as PD
import geopandas as GPD
import pickle as PKL
import rasterio as RAST
from rasterio.transform import from_origin
from shapely.geometry import Point

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kriging') )
from variogram import *
from kriging_ordinary import *

## loading data
date='2018-12'
print('Loading data %s....'% date)
df = PD.read_csv('/Volumes/Data2/Workspace/GENM/results/projectionOfRoute/routeProjection_'+date+'_3826.csv')
#df = PD.read_csv('routeProjection_'+date+'_3826.csv')

## variogram
lagrange = 25 # meter
lagmax = 5000 # meter
batchsize = 500
#distance='euclidean'
distance='cityblock'
model='exponential'

## kringning
n_neighbor=10

## Raster
LIMIT_ELEMENTS=1000
TW_XMAX=355210.
TW_XMIN=145541.
TW_YMAX=2800785.
TW_YMIN=2421543.
TW_CELLWIDTH=50

## Training model
print('Running ordinary-kriging....')
OK = kriging_ordinary( distance_type=distance, 
                       lag_range=lagrange,
                       lag_max=lagmax,
                       variogram_model=model,
                       n_jobs=len(df)/batchsize,
                       tqdm=True,
                       debug=True)
OK.fit(df[['x','y']].values, df['passingby_user'].values)
OK.update_debug(False)
OK.update_tqdm(False)
OK.variogram.plot('variogram_'+date+'_'+model+'_'+distance+'.png', show=False, title='Sill %.2f, ragne %.2f, nuget %.2f'%( OK.variogram.get_params()[0], 
                                                                                                                           OK.variogram.get_params()[1], 
                                                                                                                           OK.variogram.get_params()[2]))
PKL.dump(OK, open('model_passingby_'+date+'_'+distance+'_'+str(n_neighbor)+'.pkl', 'wb'))

