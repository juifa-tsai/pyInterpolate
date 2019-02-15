import numpy as NP

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

            - sin(2*[5])   cos(2*[5])
        c = ------------ + ----------
               4*[3]^2      4*[4]^2
        ref : https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """

    def __init__(self, radius_x=2.5, radius_y=2.5, theta=0):
        self._size0 = 0
        self.update_radiusX(radius_x)
        self.update_radiusY(radius_y)
        self.update_theta(theta)

    def update_radiusX(self, radius_x):
        if self._size0 > 0 and self._size0 != len(NP.atleast_1d(radius_x)):
            print('[ERROR] update_radiusX: input different size with fitting data')
            return
        self._radius_x = NP.atleast_1d(radius_x)

    def update_radiusY(self, radius_y):
        if self._size0 > 0 and self._size0 != len(NP.atleast_1d(radius_y)):
            print('[ERROR] update_radiusY: input different size with fitting data')
            return
        self._radius_y = NP.atleast_1d(radius_y)

    def update_theta(self, theta):
        if self._size0 > 0 and self._size0 != len(NP.atleast_1d(theta)):
            print('[ERROR] update_theta: input different size with fitting data')
            return
        self._theta = NP.atleast_1d(theta)

    def predict(self, X):
        x = NP.atleast_2d(X)[:,0][:, NP.newaxis]
        y = NP.atleast_2d(X)[:,1][:, NP.newaxis]
        size = len(x)

        x0 = NP.ones((size,self._size0))*self._x0
        y0 = NP.ones((size,self._size0))*self._y0
        z0 = NP.ones((size,self._size0))*self._z0
        a = NP.ones((size,self._size0))*self._a
        b = NP.ones((size,self._size0))*self._b
        c = NP.ones((size,self._size0))*self._c

        z =  z0 * NP.exp(-1 * (a*(x-x0)**2 + b*(y-y0)**2 + 2*c*(x-x0)*(y-y0)))
        return z.sum(axis=1)


    def fit(self, X, y):
        self._x0 = NP.atleast_2d(X)[:,0]
        self._y0 = NP.atleast_2d(X)[:,1]
        self._z0 = NP.atleast_1d(y)
        if len(self._x0) != len(self._z0):
            print('[ERROR] fit: input different size')
            return
        else:
            self._size0 = len(self._z0)

        self._a =  NP.cos(self._theta)**2/(2*self._radius_x**2) + NP.sin(self._theta)**2/(2*self._radius_y**2)
        self._b =  NP.sin(self._theta)**2/(2*self._radius_x**2) + NP.cos(self._theta)**2/(2*self._radius_y**2) 
        self._c = -NP.sin(2*self._theta)/(4*self._radius_x**2)  + NP.sin(2*self._theta)/(4*self._radius_y**2)

