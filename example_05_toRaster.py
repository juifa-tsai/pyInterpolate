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
DATE = '2016-12'
## * kringning
N_NEIGHBOR=10
RANGE_SCALE = 1
## * Raster
LIMIT_ELEMENTS=1000
TW_XMAX=355210.
TW_XMIN=145541.
TW_YMAX=2800785.
#TW_YMIN=2421543.
TW_YMIN=2721543.
TW_CELLWIDTH=50

## 
RBF = PKL.load(open('/Volumes/Data2/Workspace/GENM/results/model/interpolation_model_sticky_expected_user_2016-11_x1625_y1625_r0.pkl', 'rb'))
RBF.to_2D( TW_XMIN, TW_XMAX, TW_YMIN, TW_YMAX, TW_CELLWIDTH, tqdm=True, to_raster='./test.tif' )


### * loading model 
#OK = PKL.load(open('model_passingby_'+DATE+'_euclidean_10.pkl', 'rb'))
##OK = PKL.load(open('model_passingby_'+DATE+'_cityblock_10.pkl', 'rb'))
#
#OK.to_2D( TW_XMIN, TW_XMAX, TW_YMIN, TW_YMAX, TW_CELLWIDTH, 10, OK.variogram().get_params()[1], tqdm=True, to_raster='./test.tif' )
