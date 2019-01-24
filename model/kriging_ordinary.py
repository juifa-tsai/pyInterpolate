import numpy as NP
import scipy.linalg
from tqdm import tqdm as TQDM
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from variogram import variogram as VAR
from utility import get_neighors_brutal

algorithms = ['kdtree', 'brutal']

class kriging_ordinary(VAR):

    def __init__(self, variogram=None, n_neighbor=5, radius=NP.inf, distance_type='euclidean', lag_range=None, lag_max=NP.inf, variogram_model='poly1', variogram_params=None, variogram_paramsbound=None, n_jobs=1, variogram_good_lowbin_fit=False, tqdm=False, debug=False, algorithm='kdtree'):
        self.variogram = variogram
        self.tqdm = tqdm
        self.debug = debug
        self.good_lowbin_fit = variogram_good_lowbin_fit
        self.lags = NP.array([])
        self.variances = NP.array([])
        self.update_n_neighbor(n_neighbor)
        self.update_radius(radius)
        self.update_n_jobs(n_jobs)
        self.update_algorithm(algorithm)
        self.update_model(variogram_model)
        self.update_params(variogram_params, variogram_paramsbound)
        self.update_lag_range(lag_range)
        self.update_lag_max(lag_max)
        self.update_distance_type(distance_type)

    def update_algorithm(self, algorithm):
        if algorithm.lower() in algorithms:
            self.algorithm = algorithm
        else:
            print('[WARNING] %d not found in algorithm list'% algorithm)
            print('          list: ', algorithms)
            self.algorithm = 'kdtree'

    def update_n_neighbor(self, n_neighbor):
        self.n_neighbor = n_neighbor

    def update_radius(self, radius):
        if radius <= 0:
            print('[ERROR] radius must be > 0')
            self.radius = NP.inf
        else:
            self.radius = radius

    def update_variogram(self, variogram):
        self.variogram = variogram
        

    def fit(self, X, Y, to=None, transparent=True, show=False):
        self.Y = NP.atleast_1d(Y)
        if self.algorithm == 'kdtree':
            self.X = cKDTree(NP.atleast_1d(X), copy_data=True)
        else:
            self.X = X
        if self.variogram is None:
            print('[INFO] creating variogram....')
            self.variogram = VAR(lag_range=self.lag_range, 
                                 lag_max=self.lag_max, 
                                 distance_type=self.distance_type, 
                                 model=self.model_name, 
                                 model_params=self.params, 
                                 model_paramsbound=self.params_bound, 
                                 n_jobs=self.n_jobs, 
                                 good_lowbin_fit=self.good_lowbin_fit, 
                                 tqdm=self.tqdm,
                                 debug=self.debug)
            self.variogram.fit(X,Y)
            if self.debug:
                 print('Sill %.2f, ragne %.2f, nuget %.2f'%( self.variogram.get_params()[0], 
                                                             self.variogram.get_params()[1], 
                                                             self.variogram.get_params()[2]))
        else:
            if self.debug:
                print('[INFO] kriging_ordinary::fit(): do nothing due to variogram existed')
            else:
                pass
        self.variogram.plot(to=to, transparent=transparent, show=show)

    def predict(self, X, n_neighbor=None, radius=None):
        ''' 
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
        '''
        if n_neighbor is not None:
            self.n_neighbor = n_neighbor
        if radius is not None:
            self.radius = radius

        ## Find the neighbors 
        if self.algorithm == 'kdtree':
            print('[INFO] Finding closest neighbors with kdtree....')
            if self.distance_type == 'cityblock':
                neighbor_dst, neighbor_idx = self.X.query(NP.atleast_1d(X), k=self.n_neighbor, p=1 )
            else:
                neighbor_dst, neighbor_idx = self.X.query(NP.atleast_1d(X), k=self.n_neighbor, p=2 )
        else:
            print('[INFO] Finding closest neighbors with brutal loops....')
            neighbor_dst, neighbor_idx = get_neighors_brutal( NP.atleast_1d(X), self.X, k=self.n_neighbor, distance=self.distance, n_jobs=self.n_jobs, tqdm=self.tqdm)

        ## Calculate prediction
        print('[INFO] calculation prediction for %d data....'% len(X))
        if self.tqdm:
            batches = TQDM(range(0, len(X)))
        else:
            batches = range(0, len(X))

        results = NP.zeros(len(X))
        errors = NP.zeros(len(X))
        for nd, ni, i in zip(neighbor_dst, neighbor_idx, batches):
            if self.tqdm:
                batches.set_description(">> ")

            ni = ni[nd < self.radius]
            nd = nd[nd < self.radius]

            if len(ni) == 0: 
                continue 
            else: 
                n = len(ni)

            ## Initialization
            a = NP.zeros((n+1,n+1))  
            b = NP.ones((n+1, 1))

            ## Fill matrix a
            if self.algorithm == 'kdtree': 
                D = cdist(self.X.data[ni], self.X.data[ni], metric=self.distance_type)
            else:
                D = cdist(self.X[ni], self.X[ni], metric=self.distance_type)
            a[:n, :n] = self.variogram.predict(D)
            a[:, n] = 1
            a[n, :] = 1
            NP.fill_diagonal(a, 0)

            ## Fill vector b
            b[:-1, 0] = self.variogram.predict(nd)
            if NP.any(NP.absolute(nd) == 0 ): b[0, 0] = 0. 

            ## Get weights
            w = scipy.linalg.solve(a, b)

            ## Fill results
            results[i] = w[:n, 0].dot(self.Y[ni])
            errors[i] = w[:, 0].dot(b[:, 0])

        if self.tqdm: 
            batches.close()

        return results, errors
            

