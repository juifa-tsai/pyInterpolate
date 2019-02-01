import sys, os,time
import numpy as NP
import pandas as PD
import geopandas as GPD
import pickle as PKL
import rasterio as RAST
from rasterio.transform import from_origin
from shapely.geometry import Point

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model') )
from variogram import *
from kriging_ordinary import *

## pars
date = '2018-12'
## kringning
n_neighbor=10
range_scale = 1
## Raster
LIMIT_ELEMENTS=1000
TW_XMAX=355210.
TW_XMIN=145541.
TW_YMAX=2800785.
TW_YMIN=2421543.
TW_CELLWIDTH=50

## loading model 
#OK = PKL.load(open('model_passingby_'+date+'_euclidean_10.pkl', 'rb'))
OK = PKL.load(open('model_passingby_'+date+'_cityblock_10.pkl', 'rb'))

## Interpolation
X, Y = NP.meshgrid( NP.arange(TW_XMIN, TW_XMAX+TW_CELLWIDTH, TW_CELLWIDTH),
                    NP.arange(TW_YMIN, TW_YMAX+TW_CELLWIDTH, TW_CELLWIDTH))
xy = NP.concatenate(( X.flatten()[:, NP.newaxis], 
                      Y.flatten()[:, NP.newaxis]), 
                    axis=1)
xbins = X.shape[1]
ybins = Y.shape[0]
z = NP.array([])
ze = NP.array([])

# Interpolation with batch
print(len(xy))
print(len(xy)/LIMIT_ELEMENTS)
batches = create_batchrange( len(xy), int(len(xy)/LIMIT_ELEMENTS) )

print(">> [INFO] Interpolating %d pixels (%d, %d) to %d batches:"%(len(xy), xbins, ybins, len(batches)))
print('>>        n_neighbor: %d, radius : %.2f (m)'%(n_neighbor, OK.variogram.get_params()[1]*5))
batches= TQDM(batches)
for idx in batches:
    batches.set_description(">> ")
    z_new, ze_new = OK.predict( xy[idx[0]:idx[1], :], n_neighbor=n_neighbor, radius=OK.variogram.get_params()[1]*1 )
    z = NP.append(z, z_new)
    ze = NP.append(ze, ze_new)

# Set raster left-top coordinates
start_time = time.time()
outpath = os.path.join('./', 'raster'+str(TW_CELLWIDTH)+'_passingby_'+date+'_'+str(n_neighbor)+'_'+str(int(OK.variogram.get_params()[1]*range_scale))+'.tif')
print('>> [INFO] Outputing %s... '%(outpath))
transform = from_origin(xy[:,0].min(), xy[:,1].max(), TW_CELLWIDTH, TW_CELLWIDTH)
raster = RAST.open( outpath, 'w',
                    driver='GTiff',
                    height=ybins,
                    width=xbins,
                    count=1,
                    dtype=z.dtype,
                    crs='epgs:3826',
                    transform=transform )
raster.write(NP.flip(z.reshape(ybins,xbins), 0), 1)
raster.close()

outpath = os.path.join('./', 'raster'+str(TW_CELLWIDTH)+'_error_passingby_'+date+'_'+str(n_neighbor)+'_'+str(int(OK.variogram.get_params()[1]*range_scale))+'.tif')
print('>> [INFO] Outputing %s... '%(outpath))
transform = from_origin(xy[:,0].min(), xy[:,1].max(), TW_CELLWIDTH, TW_CELLWIDTH)
raster = RAST.open( outpath, 'w',
                    driver='GTiff',
                    height=ybins,
                    width=xbins,
                    count=1,
                    dtype=z.dtype,
                    crs='epgs:3826',
                    transform=transform )
raster.write(NP.flip(ze.reshape(ybins,xbins), 0), 1)
raster.close()

print(">> [DONE] --- %.2f seconds ---" % float(time.time() - start_time))
print(">> ----------------------------------")

