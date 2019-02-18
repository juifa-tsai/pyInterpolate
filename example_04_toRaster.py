import sys, os,time
import numpy as NP
import pandas as PD
import geopandas as GPD
import pickle as PKL
import rasterio as RAST
from rasterio.transform import from_origin
from shapely.geometry import Point
from kriging.variogram import *
from kriging.kriging_ordinary import *

## * pars
DATE = '2018-12'
## * kringning
N_NEIGHBOR=10
RANGE_SCALE = 1
## * Raster
LIMIT_ELEMENTS=1000
TW_XMAX=355210.
TW_XMIN=145541.
TW_YMAX=2800785.
TW_YMIN=2421543.
TW_CELLWIDTH=50

## * loading model 
#OK = PKL.load(open('model_passingby_'+DATE+'_euclidean_10.pkl', 'rb'))
OK = PKL.load(open('model_passingby_'+DATE+'_cityblock_10.pkl', 'rb'))

## * Interpolation to Raster
### 1. Create grid of extent
X, Y = NP.meshgrid( NP.arange(TW_XMIN, TW_XMAX+TW_CELLWIDTH, TW_CELLWIDTH),
                    NP.arange(TW_YMIN, TW_YMAX+TW_CELLWIDTH, TW_CELLWIDTH))
xy = NP.concatenate(( X.flatten()[:, NP.newaxis], 
                      Y.flatten()[:, NP.newaxis]), 
                    axis=1)
xbins = X.shape[1]
ybins = Y.shape[0]
z = NP.array([])
ze = NP.array([])

### 2. Interpolation with batch to save memory
def create_batchrange( n_data, n_jobs ):
    '''
    return index range [[0,5], [5,10], ... ]
    '''
    size = int(n_data/n_jobs)
    idxs = [size*n for n in range(0,n_jobs)]
    idxs.append(n_data)
    return [[idxs[n], idxs[n+1]] for n in range(n_jobs)]

print(len(xy))
print(len(xy)/LIMIT_ELEMENTS)
batches = create_batchrange( len(xy), int(len(xy)/LIMIT_ELEMENTS) )

print(">> [INFO] Interpolating %d pixels (%d, %d) to %d batches:"%(len(xy), xbins, ybins, len(batches)))
print('>>        n_neighbor: %d, radius : %.2f (m)'%(N_NEIGHBOR, OK.variogram().get_params()[1]*5))
batches= TQDM(batches)
for idx in batches:
    batches.set_description(">> ")
    z_new, ze_new = OK.predict( X = xy[idx[0]:idx[1], :], 
                                n_neighbor = N_NEIGHBOR, 
                                radius = OK.variogram().get_params()[1]*RANGE_SCALE 
                              )
    z = NP.append(z, z_new)
    ze = NP.append(ze, ze_new)

### 3. Output raster
start_time = time.time()

#### Passingby-user raster
outpath = os.path.join('./', 'raster'+str(TW_CELLWIDTH)+'_passingby_'+DATE+'_'+str(N_NEIGHBOR)+'_'+str(int(OK.variogram().get_params()[1]*RANGE_SCALE))+'.tif')
print('>> [INFO] Outputing to %s... '%(outpath))

transform = from_origin(xy[:,0].min(), xy[:,1].max(), TW_CELLWIDTH, TW_CELLWIDTH) # Set raster left-top's coordinates
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

##### Error of passingby-user raster
outpath = os.path.join('./', 'raster'+str(TW_CELLWIDTH)+'_error_passingby_'+DATE+'_'+str(N_NEIGHBOR)+'_'+str(int(OK.variogram.get_params()[1]*RANGE_SCALE))+'.tif')
print('>> [INFO] Outputing to %s... '%(outpath))

transform = from_origin(xy[:,0].min(), xy[:,1].max(), TW_CELLWIDTH, TW_CELLWIDTH) # Set raster left-top's coordinates
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

