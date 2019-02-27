import numpy as NP
import rasterio as RAST
from tqdm import tqdm as TQDM
from rasterio.transform import from_origin
from .utility import create_batchrange

class gaussian:
    """2D gaussian pars[6] = [x0, y0, amp, radius_x, radius_y, theta]
    
                                        2               2             
                         -[ a*(x - [0])^  + b*(y - [1])^  + c*(x - [0])*(y - [1]) ]
        y(x,y) = [2] * e^

        where a, b and c are

            cos([5])^2   sin([5])^2
        a = ---------- + ----------
             2*[3]^2      2*[4]^2

            sin([5])^2   cos([5])^2
        b = ---------- + ----------
             2*[3]^2      2*[4]^2

            - sin(2*[5])   sin(2*[5])
        c = ------------ + ----------
               4*[3]^2      4*[4]^2
        ref : https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """

    def __init__(self, radius_x=2.5, radius_y=2.5, theta=0, debug=False):
        self.debug = debug
        self._size=0
        self.update_params( radius_x, radius_y, theta )

    def update_params(self, radius_x, radius_y, theta):
        if len(NP.atleast_1d(radius_x)) != len(NP.atleast_1d(radius_y)) or len(NP.atleast_1d(radius_x)) != len(NP.atleast_1d(theta)):
            print('>> [ERROR] update_params: input different size')
            return
        else:
            self._radius_x = NP.atleast_1d(radius_x)
            self._radius_y = NP.atleast_1d(radius_y)
            self._theta = NP.atleast_1d(theta)
            self._size_params = len(self._theta)

    def fit(self, X, y):
        self._x0 = NP.atleast_2d(X)[:,0]
        self._y0 = NP.atleast_2d(X)[:,1]
        self._z0 = NP.atleast_1d(y)
        if len(NP.atleast_2d(X)[:,0]) != len(NP.atleast_1d(y)):
            print('>> [ERROR] fit: input different size')
            return
        elif self._size_params > 1 and self._size_params != len(NP.atleast_1d(y)):
            print('>> [ERROR] fit: input different size with parameters')
            return
        else:
            self._x0 = NP.atleast_2d(X)[:,0]
            self._y0 = NP.atleast_2d(X)[:,1]
            self._z0 = NP.atleast_1d(y)
            self._size = len(self._z0)

        self._a =  NP.cos(self._theta)**2/(2*self._radius_x**2) + NP.sin(self._theta)**2/(2*self._radius_y**2)
        self._b =  NP.sin(self._theta)**2/(2*self._radius_x**2) + NP.cos(self._theta)**2/(2*self._radius_y**2) 
        self._c = -NP.sin(2*self._theta)/(4*self._radius_x**2)  + NP.sin(2*self._theta)/(4*self._radius_y**2)

    def predict(self, X):
        x = NP.atleast_2d(X)[:,0][:, NP.newaxis]
        y = NP.atleast_2d(X)[:,1][:, NP.newaxis]
        size = len(x)

        x0 = NP.ones((size,self._size))*self._x0
        y0 = NP.ones((size,self._size))*self._y0
        z0 = NP.ones((size,self._size))*self._z0
        a = NP.ones((size,self._size))*self._a
        b = NP.ones((size,self._size))*self._b
        c = NP.ones((size,self._size))*self._c

        z =  z0 * NP.exp(-1 * (a*(x-x0)**2 + b*(y-y0)**2 + 2*c*(x-x0)*(y-y0)))
        return z.sum(axis=1)

    def to_2D(self, xmin, xmax, ymin, ymax, cellsize=50, n_cell_per_job=1000, tqdm=False, crs='epsg:3826', to_raster=None, ):
        """
        [DESCRIPTION]
            Output raster data of prediction w.r.t. given extent size
        [INPUT]
            xmin      : float,  minimum value of x-axis of extent (None) 
            xmax      : float,  maximum value of x-axis of extent (None) 
            ymin      : float,  minimum value of y-axis of extent (None)
            ymax      : float,  maximum value of y-axis of extent (None)
            cellsize  : flaot,  cell size of extent (50)
            n_cell_per_job : int, number of cell in processing batch jobs (1000)
            tqdm      : bool,   if show the tqdm processing bar (False)
            crs       : string, the projection code for output raster tif ('epsg:3826')
            to_raster : string, path to save raster tif. If None, the function return the values only (None) 
        [OUTPUT]
            predicted value,
            grid of extent
        """
        if self._size == 0:
            print(">> [ERROR] do fit() before calling to_2D()")
            return

        ## Create 2D extent
        ### Grid is from top-left to bottom-right
        X, Y = NP.meshgrid( NP.arange(xmin, xmax+cellsize, cellsize),
                            NP.arange(ymin, ymax+cellsize, cellsize))
        ### set prediction location is in the central of grid
        xy = NP.concatenate(( X.flatten()[:, NP.newaxis] + cellsize/2, 
                              Y.flatten()[:, NP.newaxis] - cellsize/2), 
                            axis=1)

        ## Define variables
        xbins = X.shape[1]
        ybins = Y.shape[0]
        z = NP.array([])

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
            z_new = self.predict(xy[idx[0]:idx[1], :])
            z = NP.append(z, z_new)

        ## Set raster left-top's coordinates
        if to_raster is not None:
            ### Set raster left-top's coordinates
            transform = from_origin(X.min(), Y.max(), cellsize, cellsize)

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
        return z, xy

