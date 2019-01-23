import sys, os
import numpy as NP
import pandas as PD
import geopandas as GPD
from shapely.geometry import Point

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model') )
from variogram import *
from kriging_ordinary import *

print('Loading data....')
df = PD.read_csv('/Volumes/Data2/Workspace/GENM/results/projectionOfRoute/routeProjection_2016-01_3826.csv')

## data
test_size = 0.9
## variogram
lagrange = 25 # meter
lagmax = 2000 # meter
batchsize = 500
distance='euclidean'
#distance='cityblock'
model='exponential'
## kringning
n_neighbor=3

NP.random.seed(0)
a = NP.arange(len(df))
NP.random.shuffle(a)
test_indexes = a[:int(NP.round(len(df)*test_size))]
train_indexes = [index for index in df.index if index not in test_indexes]
df_test = df.loc[test_indexes,:].copy()
df_train = df.loc[train_indexes,:].copy()
print('Number of observations in training: {}, in test: {}'.format(len(df_train), len(df_test)))

print('Running ordinary-kriging....')
ok = kriging_ordinary( distance_type=distance, 
                       lag_range=lagrange,
                       lag_max=lagmax,
                       variogram_model=model,
                       n_jobs=len(df_train)/batchsize,
                       tqdm=True,
                       debug=True)
ok.fit(df_train[['x','y']].values, df_train['passingby_user'].values, to='variogram.png')

df_train['prediction'] = ok.predict( df_train[['x','y']].values, n_neighbor=n_neighbor, radius=ok.variogram.get_params()[1]*3)[0]
df_test['prediction']  = ok.predict( df_test[['x','y']].values, n_neighbor=n_neighbor, radius=ok.variogram.get_params()[1]*3)[0]
df_train['kriging_residual'] = df_train['passingby_user'] - df_train['prediction']
df_test['kriging_residual'] = df_test['passingby_user'] - df_test['prediction']


## Ploting
import matplotlib.pyplot as plt
print('Ploting results')
plt.figure(figsize=(6,6))
plt.subplot(221)
plt.plot(df_train['prediction'], df_train['passingby_user'], '.')
plt.title('Training: pred vs obs')
plt.xlabel('Predictions')
plt.ylabel('True value')
plt.plot([0,100], [0,100], 'g--')
plt.ylim(0,100)
plt.xlim(0,100)
plt.subplot(222)
df_train['kriging_residual'].hist()
plt.title('Hist training res\nMedian absolute error: {:.1f}'.format(NP.median(NP.abs(df_train['kriging_residual']))))
plt.subplot(223)
plt.plot(df_test['prediction'], df_test['passingby_user'], '.')
plt.plot([0,100], [0,100], 'g--')
plt.title('Test: pred vs obs')
plt.xlabel('Predictions')
plt.ylabel('True value')
plt.ylim(0,100)
plt.xlim(0,100)
plt.subplot(224)
df_test['kriging_residual'].hist()
plt.title('Hist test res\nMedian absolute error: {:.1f}'.format(NP.median(NP.abs(df_test['kriging_residual']))))
plt.tight_layout()
plt.savefig('prediction.png')
