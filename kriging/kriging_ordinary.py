import numpy as NP
import scipy.linalg
import matplotlib.pyplot as PLT
import rasterio as RAST
from rasterio.transform import from_origin
from shapely.geometry import Point
from tqdm import tqdm as TQDM
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from .variogram import variogram as VAR
from .utility import get_neighors_brutal, create_batchrange

algorithms = ['kdtree', 'brutal']

class kriging_ordinary(VAR):

    def __init__(self, variogram=None, distance_type='euclidean', lag_range=None, lag_max=NP.inf, variogram_model='poly1', variogram_params=None, variogram_paramsbound=None, n_jobs=1, variogram_good_lowbin_fit=False, tqdm=False, debug=False, algorithm='kdtree'):
    
        self._X = None
        self.n_features = 0
        self.params = variogram_params 
        self.params_bound = variogram_paramsbound
        self.update_variogram(variogram)
        self.update_good_lowbin_fit(variogram_good_lowbin_fit)
        self.update_tqdm(tqdm)
        self.update_debug(debug)
        self.update_n_jobs(n_jobs)
        self.update_algorithm(algorithm)
        self.update_model(variogram_model)
        self.update_lag_range(lag_range)
        self.update_lag_max(lag_max)
        self.update_distance_type(distance_type)

    def update_good_lowbin_fit(self, good_lowbin_fit):
        '''
        [DESCRIPTION]
            Update good_lowbin_fit option to turn on/off to improve the fitting in lowed bins.
            The model has to be re-fit by calling update_fit() after updated variogram.
        [INPUT]
            good_lowbin_fit : bool 
        [OUTPUT]
            Null
        '''
        self.good_lowbin_fit = good_lowbin_fit

    def update_tqdm(self, tqdm):
        '''
        [DESCRIPTION]
            Turn on/off to runing process bar.
        [INPUT]
            tqdm : bool 
        [OUTPUT]
            Null
        '''
        self.tqdm = tqdm

    def update_debug(self, debug):
        '''
        [DESCRIPTION]
            Turn on/off to debug.
        [INPUT]
            debug : bool 
        [OUTPUT]
            Null
        '''

        self.debug = debug

    def update_algorithm(self, algorithm):
        '''
        [DESCRIPTION]
            update algorithm of data structure
        [INPUT]
            algorithm : string, algorithm name of data structure %s
        [OUTPUT]
            Null
        '''%( algorithms )
        if algorithm.lower() in algorithms:
            self._algorithm = algorithm
        else:
            print('>> [WARNING] %d not found in algorithm list'% algorithm)
            print('             list: ', algorithms)
            self._algorithm = 'kdtree'

    def update_variogram(self, variogram):
        '''
        [DESCRIPTION]
            Update variogram model. The model has to be re-fit by calling update_fit() after updated variogram.
        [INPUT]
            variogram : callable, input variogram model
        [OUTPUT]
            Null
        '''
        self._variogram = variogram

    def update_fit(self, X=None, y=None, to=None, transparent=True, show=False):
        '''
        [DESCRIPTION]
            Update the fitting. If X and y are not None, it stores the data with KDTree or original data structure, 
            and it re-fit the variogram model with set parameters. Otherwise, it do re-fit only. 
        [INPUT]
            X          : array-like, input data with features, same data size with y. (None)
            y          : array-like, input data with interesting value, same data size with X. (None)
            to         : string,     path to save plot (None) 
            transparent: True,       transparent background of plot (True)
            show       : bool,       if show on screen (True)
        [OUTPUT]
            Null
        '''
        if X is not None:
            if self._algorithm == 'kdtree':
                self._X = cKDTree(X, copy_data=True)
            else:
                self._X = X
        if y is not None: 
            self.y = y

        if self._algorithm == 'kdtree': 
            self.fit(self._X.data, self.y, to=to, transparent=transparent, show=show)
        else:
            self.fit(self._X, self.y, to=to, transparent=transparent, show=show)

    def fit(self, X, y, to=None, transparent=True, show=False):
        '''
        [DESCRIPTION]
            Store the data with KDTree or original data structure, and fit the variogram model with set parameters. 
        [INPUT]
            X          : array-like, input data with features, same data size with y.
            y          : array-like, input data with interesting value, same data size with X
            to         : string,     path to save plot (None) 
            transparent: True,       transparent background of plot (True)
            show       : bool,       if show on screen (True)
        [OUTPUT]
            Null
        '''
        X = NP.atleast_1d(X)
        y = NP.atleast_1d(y)
        if len(X) != len(y):
            print(">> [ERROR] different size %di, %d"%(len(X), len(y)))
            raise
        if self._variogram is None:
            print('>> [INFO] creating variogram....')
            self._variogram = VAR(lag_range=self.lag_range, 
                                  lag_max=self.lag_max, 
                                  distance_type=self.distance_type, 
                                  model=self.model_name, 
                                  model_params=self.params, 
                                  model_paramsbound=self.params_bound, 
                                  n_jobs=self.n_jobs, 
                                  good_lowbin_fit=self.good_lowbin_fit, 
                                  tqdm=self.tqdm,
                                  debug=self.debug)
            self._variogram.fit(X,y)
            if self.debug:
                print('>> [DONE] Sill %.2f, ragne %.2f, nuget %.2f'%( self._variogram.get_params()[0], 
                                                                      self._variogram.get_params()[1], 
                                                                      self._variogram.get_params()[2]))
        else:
            if self.debug:
                print('>> [INFO] kriging_ordinary::fit(): do nothing due to variogram existed')
            else:
                pass

        self.y = y
        if self._algorithm == 'kdtree':
            self._X = cKDTree(X, copy_data=True)
        else:
            self._X = X

        if len(NP.shape(X)) == 1:
            self.n_features = 1
        else:
            self.n_features = NP.shape(X)[1]
        
        if not show or to is not None:
            self._variogram.plot(to=to, transparent=transparent, show=show)

    def variogram(self):
        '''
        [DESCRIPTION]
            call the variogram model.
        [INPUT]
            Null
        [OUTPUT]
            callable, variogram model
        '''
        return self._variogram

    def predict(self, X, n_neighbor=5, radius=NP.inf, use_nugget=False, get_error=False):
        '''
        [DESCRIPTION]
           Calculate and predict interesting value with looping for all data, the method take long time but save memory
           Obtain the linear argibra terms
            | V_ij 1 || w_i | = | V_k |
            |  1   0 ||  u  |   |  1  |
                a    *   w    =    b
              w = a^-1 * b
              y = w_i * Y
            V_ij : semi-variance matrix within n neighors
            V_k  : semi-variance vector between interesting point and n neighbors
            w_i  : weights for linear combination
            u    : lagrainge multiplier
            Y    : true value of neighbors
            y    : predicted value of interesting point
        [INPUT]
            X          : array-like, input data with same number of fearture in training data
            n_neighbor : int,        number of neighbor w.r.t input data, while distance < searching radius (5)
            radius     : float,      searching radius w.r.t input data (inf)
            use_nugget : bool,       if use nugget to be diagonal of kriging matrix for prediction calculation (False)
            get_error  : bool,       if return error (False)
        [OUTPUT]
            1D/2D array(float)
            prediction : float, prdicted value via Kriging system
            error      : float, error of predicted value (only if get_error = True)
        '''
        X = NP.atleast_1d(X)
        if len(NP.shape(X)) == 1:
            if self.n_features == len(X):
                X = NP.atleast_2d(X)
            elif self.n_features != 1:
                print('>> [ERROR] wrong input number of features %d, %d'%( 1, self.n_features))
                return
        elif NP.shape(X)[1] != self.n_features:
            print('>> [ERROR] wrong input number of features %d, %d'%( NP.shape(X)[1], self.n_features))
            return

        if n_neighbor <= 0:
            print(">> [ERROR] number of neighbor must be > 0")
            return

        if radius <= 0:
            print('>> [ERROR] radius must be > 0')
            return

        ## Find the neighbors 
        if self._algorithm == 'kdtree':
            if self.debug:
                print('>> [INFO] Finding closest neighbors with kdtree....')
            if self.distance_type == 'cityblock':
                neighbor_dst, neighbor_idx = self._X.query(X, k=n_neighbor, p=1 )
            else:
                neighbor_dst, neighbor_idx = self._X.query(X, k=n_neighbor, p=2 )
        else:
            if self.debug:
                print('>> [INFO] Finding closest neighbors with brutal loops....')
            neighbor_dst, neighbor_idx = get_neighors_brutal( X, self._X, k=n_neighbor, distance=self.distance, n_jobs=self.n_jobs, tqdm=self.tqdm)

        ## Calculate prediction
        if self.debug:
            print('>> [INFO] calculation prediction for %d data....'% len(X))
        if self.tqdm and not self.debug:
            batches = TQDM(range(0, len(X)))
        else:
            batches = range(0, len(X))

        results = NP.zeros(len(X))
        errors = NP.zeros(len(X))
        for nd, ni, i in zip(neighbor_dst, neighbor_idx, batches):
            if self.tqdm and not self.debug:
                batches.set_description(">> ")
            elif self.debug:
                print(">> [%d] %d neighbors distance : %s "%(i, len(ni), str(nd)))
            
            ni = ni[nd < radius] # neighbors' index, while the distance < search radius
            nd = nd[nd < radius] # neighbors' distance, while the distance < search radius

            if len(ni) == 0: 
                continue 
            else: 
                n = len(ni)

            ## Initialization
            a = NP.zeros((n+1,n+1))  
            b = NP.ones((n+1, 1))

            ## Fill matrix a
            if self._algorithm == 'kdtree': 
                D = cdist(self._X.data[ni], self._X.data[ni], metric=self.distance_type)
            else:
                D = cdist(self._X[ni], self._X[ni], metric=self.distance_type)
            a[:n, :n] = self._variogram.predict(D)
            a[:, n] = 1
            a[n, :] = 1
            a[n, n] = 0

            ## Fill vector b
            b[:-1, 0] = self._variogram.predict(nd)

            ## set self-varinace is zero if not using Nugget
            if not use_nugget:
                ## modify a
                NP.fill_diagonal(a, 0.)
                ## modify b
                zero_index = NP.where(NP.absolute(nd) == 0)
                if len(zero_index) > 0:
                    b[zero_index[0], 0] = 0.
            elif self.debug:
                print(">>      Turn on using Nugget for the diagonal of kriging matrix") 



            ## Get weights
            w = scipy.linalg.solve(a, b)

            if self.debug:
                print(">>      %d neighbors < %.2f distance : %s "%(len(ni), radius, str(nd)))
                print(">>      Fitted kriging matrix a: ")
                print(a)
                print(">>      Fitted semivarince between the location to %d neighbors b: "% n_neighbor)
                print(b)
                print(">>      Weights w: ")
                print(w)
                print(">>      Observe value y: ")
                print(self.y[ni])

            ## Fill results
            results[i] = w[:n, 0].dot(self.y[ni])
            errors[i] = w[:, 0].dot(b[:, 0])

        if self.tqdm and not self.debug: 
            batches.close()

        if get_error:
            return results, errors
        else:
            return results

            
    def plot(self, to=None, title='', transparent=True, show=True):
        """
        [DESCRIPTION]
            Displays or draw variogram model with the actual binned data.
        [INPUT]
            to          : string, path to save plot (None) 
            title       : string, title in the plot ('')
            transparent : True,   transparent background of plot (True)
            show        : bool,   if show on screen (True)
        [OUTPUT]
            Null
        """
        fig = PLT.figure()
        ax = fig.add_subplot(111)
        ax.plot(self._variogram.lags, self._variogram.variances, 'r*')
        ax.plot(self._variogram.lags, self._variogram.model(self._variogram.params, self._variogram.lags), 'k-')
        PLT.title(title)
        if show:
            PLT.show()
        if to is not None:
            print('>> [INFO] Saving plot to %s'% to)
            PLT.savefig(to, transparent=transparent)

    def to_2D(self, xmin, xmax, ymin, ymax, cellsize=50, n_neighbor=5, radius=NP.inf, use_nugget=False, get_error=False, n_cell_per_job=1000, tqdm=False, crs='epgs:3826', to_raster=None, ):
        """
        [DESCRIPTION]
            Output raster data of prediction w.r.t. given extent size
        [INPUT]
            xmin      : float,  minimum value of x-axis of extent (None) 
            xmax      : float,  maximum value of x-axis of extent (None) 
            ymin      : float,  minimum value of y-axis of extent (None)
            ymax      : float,  maximum value of y-axis of extent (None)
            cellsize  : flaot,  cell size of extent (50)
            n_neighbor: int,    number of neighbors for model prediction (5)
            radius    : float,  searching radius w.r.t input data (inf)
            use_nugget: bool,   if use nugget to be diagonal of kriging matrix for prediction calculation (False)
            get_error : bool,   if return error (False)
            n_cell_per_job : int, number of cell in processing batch jobs (1000)
            tqdm      : bool,   if show the tqdm processing bar (False)
            crs       : string, the projection code for output raster tif ('epgs:3826')
            to_raster : string, path to save raster tif. If None, the function return the values only (None) 
        [OUTPUT]
            predicted value,
            (predicted error),
            grid of extent
        """
        if self._X is None:
            print(">> [ERROR] do fit() before calling to_2D()")
            return

        ## Create 2D extent
        X, Y = NP.meshgrid( NP.arange(xmin, xmax+cellsize, cellsize),
                            NP.arange(ymin, ymax+cellsize, cellsize))
        xy = NP.concatenate(( X.flatten()[:, NP.newaxis], 
                              Y.flatten()[:, NP.newaxis]), 
                            axis=1)

        ## Define variables
        xbins = X.shape[1]
        ybins = Y.shape[0]
        z = NP.array([])
        if get_error:
            ze = NP.array([])

        ## Create batches 
        batches = create_batchrange( len(xy), int(len(xy)/n_cell_per_job) )
        if self.debug: 
            print(">> [INFO] Interpolating %d pixels (%d, %d) to %d batches:"%(len(xy), xbins, ybins, len(batches)))
        if tqdm:
            batches= TQDM(batches)

        ## Predict values
        for idx in batches:
            if tqdm:
                batches.set_description(">> ")
            if get_error:
                z_new, ze_new = self.predict( X          = xy[idx[0]:idx[1], :], 
                                              n_neighbor = n_neighbor, 
                                              radius     = radius, 
                                              use_nugget = use_nugget, 
                                              get_error  = True )
                ze = NP.append(ze, ze_new)
            else:
                z_new = self.predict( X          = xy[idx[0]:idx[1], :], 
                                      n_neighbor = n_neighbor, 
                                      radius     = radius, 
                                      use_nugget = use_nugget, 
                                      get_error  = False )
            z = NP.append(z, z_new)

        ## Set raster left-top's coordinates
        if to_raster is not None:
            ### Set raster left-top's coordinates
            transform = from_origin(xy[:,0].min(), xy[:,1].max(), cellsize, cellsize)

            ### Writing raster
            raster = RAST.open( to_raster, 
                                'w',
                                driver='GTiff',
                                height=ybins,
                                width=xbins,
                                count=1,
                                dtype=z.dtype,
                                crs=crs,
                                transform=transform )
            raster.write(NP.flip(z.reshape(ybins,xbins), 0), 1)
            raster.close()

            if get_error:
                raster = RAST.open( to_raster, 
                                    'w',
                                    driver='GTiff',
                                    height=ybins,
                                    width=xbins,
                                    count=1,
                                    dtype=ze.dtype,
                                    crs=crs,
                                    transform=transform )
                raster.write(NP.flip(ze.reshape(ybins,xbins), 0), 1)
                raster.close()

        if get_error:
            return z, ze, xy
        else:
            return z, xy


