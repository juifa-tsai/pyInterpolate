import numpy as NP
import pandas as PD
import geopandas as GPD
from shapely.geometry import Point
from model.variogram import *

print('Loading data....')
df = PD.read_csv('/Volumes/Data2/Workspace/GENM/results/projectionOfRoute/routeProjection_2016-01_3826.csv')

test_size = 0.3
lagrange = 50 # meter
lagmax = 5000 # meter
batchsize = 500
distance='euclidean'
#distance='cityblock'

NP.random.seed(0)
a = NP.arange(len(df))
NP.random.shuffle(a)
test_indexes = a[:int(NP.round(len(df)*test_size))]
train_indexes = [index for index in df.index if index not in test_indexes]
df_test = df.loc[test_indexes,:].copy()
df_train = df.loc[train_indexes,:].copy()
print('Number of observations in training: {}, in test: {}'.format(len(df_train), len(df_test)))

var = variogram( lag_range=lagrange, 
                 lag_max=lagmax, 
                 distance_type=distance,
                 debug=True, 
                 n_jobs=len(df_train)/batchsize,
                 tqdm=True)

## variogram
for i, model in enumerate(['exponential', 'spherical', 'gaussian']):
    print('Model %s'% model)
    var.update_model(model)
    if i == 0:
        var.fit(df_train[['x','y']].values, df_train['passingby_user'].values)
    else:
        var.update_fit()
    var.plot('variogram_'+model+'_'+distance+'.png', show=False, title='Sill %.2f, ragne %.2f, nuget %.2f'%(var.get_params()[0], var.get_params()[1], var.get_params()[2]))
    print('Sill %.2f, ragne %.2f, nuget %.2f'%(var.get_params()[0], var.get_params()[1], var.get_params()[2]))


