import numpy as NP

#__doc__ = """
#Function definitions for variogram models. In each function, m is a list of
#defining parameters and d is an array of the distance values at which to
#calculate the variogram model.
#
#References
#----------
#.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
#    Hydrogeology, (Cambridge University Press, 1997) 272 p.
#"""

## variogram distriubtions
def variogram_poly1(m, d):
    """polynomial 1-order model, m is [slope, nugget]
        y = [1] * x + [0]
    """
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


def variogram_power(m, d):
    """Power model, m is [scale, exponent, nugget]
        y = [1] * x^[2] + [0]
    """
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d**exponent + nugget


def variogram_gaussian(m, d):
    """Gaussian model, m is [psill, range, nugget]
                              7*x    
                         - ( ----- )^2   
                             4*[2]      
        y = [1] * ( 1 - e^            ) + [0]
    """
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - NP.exp(-d**2./(range_*4./7.)**2.)) + nugget


def variogram_exponential(m, d):
    """Exponential model, m is [psill, range, nugget]
                              x     
                         - ( --- )  
                             [2]    
        y = [1] * ( 1 - e^        ) + [0]
    """
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - NP.exp(-d/(range_))) + nugget


def variogram_spherical(m, d):
    """Spherical model, m is [psill, range, nugget]
                                3 * x            x^3  
        x <= [2] : y = [1] * ( ---------  -  ---------- ) + [0]
                                2 * [2]       2 * [2]^3
        x > [2]  : y = [1] + [0]
    """
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return NP.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * ((3.*x)/(2.*range_) - (x**3.)/(2.*range_**3.)) + nugget, psill + nugget])


def variogram_hole_effect(m, d):
    """Hole Effect model, m is [psill, range, nugget]
                                          3 * x
                                     - ( ------- )   
                          3 * x            [2]
        y = [1] * ( 1 - -------- * e^             ) + [0]
                          [2]       
    """
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1. - (1.-d/(range_/3.)) * NP.exp(-d/(range_/3.))) + nugget


def variogram_circular(m, d):
    """Circular model, m is [psill, range, nugget]
                                                            --------------
                              /             2              /       x        \
        x <= [2] : y = [1] * |  1 - ----------------- +   / 1 - (-----)^2    | + [0]
                              \       pi * cos(x/[2])    V        [2]       /
        x > [2]  : y = [1] + [0]
    """
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return NP.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * (1 - 2/NP.pi/NP.cos(x/range_) + NP.sqrt(1-(x/range_)**2)) + nugget, psill + nugget])

distributions = { 'poly1':variogram_poly1, 
                  'power':variogram_power,
                  'gaussian':variogram_gaussian,
                  'exponential':variogram_exponential,
                  'spherical':variogram_spherical,
                  'hole_effect':variogram_hole_effect,
                  'circular':variogram_circular }

