import numpy as NP

#__doc__ = """
#Function definitions for variogram models. In each function, pars is a list of
#defining parameters and d is an array of the distance values at which to
#calculate the variogram model.
#
#References
#----------
#.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
#    Hydrogeology, (Cambridge University Press, 1997) 272 p.
#"""

## variogram distriubtions
def variogram_poly1(pars, d):
    """polynomial 1-order model, pars[2] = [slope, nugget]
        y = [1] * x + [0]
    """
    slope = float(pars[0])
    nugget = float(pars[1])
    return slope * d + nugget


def variogram_power(pars, d):
    """Power model, pars[3] = [scale, exponent, nugget]
        y = [1] * x^[2] + [0]
    """
    scale = float(pars[0])
    exponent = float(pars[1])
    nugget = float(pars[2])
    return scale * d**exponent + nugget


def variogram_gaussian(pars, d):
    """Gaussian model, pars[3] = [psill, range, nugget]
                              7*x    
                         - ( ----- )^2   
                             4*[2]      
        y = [1] * ( 1 - e^            ) + [0]
    """
    psill = float(pars[0])
    range_ = float(pars[1])
    nugget = float(pars[2])
    return psill * (1. - NP.exp(-d**2./(range_*4./7.)**2.)) + nugget


def variogram_exponential(pars, d):
    """Exponential model, pars[3] = [psill, range, nugget]
                              x     
                         - ( --- )  
                             [2]    
        y = [1] * ( 1 - e^        ) + [0]
    """
    psill = float(pars[0])
    range_ = float(pars[1])
    nugget = float(pars[2])
    return psill * (1. - NP.exp(-d/(range_))) + nugget


def variogram_spherical(pars, d):
    """Spherical model, pars[3] = [psill, range, nugget]
                                3 * x            x^3  
        x <= [2] : y = [1] * ( ---------  -  ---------- ) + [0]
                                2 * [2]       2 * [2]^3
        x > [2]  : y = [1] + [0]
    """
    psill = float(pars[0])
    range_ = float(pars[1])
    nugget = float(pars[2])
    return NP.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * ((3.*x)/(2.*range_) - (x**3.)/(2.*range_**3.)) + nugget, psill + nugget])


def variogram_hole_effect(pars, d):
    """Hole Effect model, pars[3] = [psill, range, nugget]
                                          3 * x
                                     - ( ------- )   
                          3 * x            [2]
        y = [1] * ( 1 - -------- * e^             ) + [0]
                          [2]       
    """
    psill = float(pars[0])
    range_ = float(pars[1])
    nugget = float(pars[2])
    return psill * (1. - (1.-d/(range_/3.)) * NP.exp(-d/(range_/3.))) + nugget


def variogram_circular(pars, d):
    """Circular model, pars[3] = [psill, range, nugget]
                                                            --------------
                              /             2              /       x        \
        x <= [2] : y = [1] * |  1 - ----------------- +   / 1 - (-----)^2    | + [0]
                              \       pi * cos(x/[2])    V        [2]       /
        x > [2]  : y = [1] + [0]
    """
    psill = float(pars[0])
    range_ = float(pars[1])
    nugget = float(pars[2])
    return NP.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * (1 - 2/NP.pi/NP.cos(x/range_) + NP.sqrt(1-(x/range_)**2)) + nugget, psill + nugget])

distributions = { 'poly1':variogram_poly1, 
                  'power':variogram_power,
                  'gaussian':variogram_gaussian,
                  'exponential':variogram_exponential,
                  'spherical':variogram_spherical,
                  'hole_effect':variogram_hole_effect,
                  'circular':variogram_circular }

