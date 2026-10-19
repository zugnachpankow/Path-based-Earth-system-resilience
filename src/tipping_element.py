"""tipping_element classes (vendored pycascades)"""
import numpy as np

class tipping_element:
    """abstract tipping_element interface; inherit from concrete classes"""

    def __init__(self):
        self._type = None
        self._par = {}

    def get_type(self):
        return self._type

    def get_par(self):
        return self._par

    def set_par(self, key, val):
        self._par[key] = val

    def dxdt_diag(self):
        return lambda t, x : 0

    def jac_diag(self):
        return lambda t, x : 0

    def tip_state(self):
        return lambda x : 0

class cusp(tipping_element):
    """cusp-like tipping element"""
    def __init__(self, a = -4, b = 1, c = 0, x_0 = 0.5 ):
        super().__init__()
        self._type = 'cusp'
        self._par['a'] = a
        self._par['b'] = b
        self._par['c'] = c
        self._par['x_0'] = x_0
        
    def dxdt_diag(self):
        return lambda t, x : self._par['a'] * pow(x - self._par['x_0'],3) \
                           + self._par['b'] * (x - self._par['x_0']) \
                           + self._par['c'](t)

    def jac_diag(self):
        return lambda t, x : 3 * self._par['a'] * pow(x - self._par['x_0'],2) \
                               + self._par['b']

    def tip_state(self):
        return lambda x : x > self._par['x_0']

class realistic_cusp(cusp):
    """cusp-like tipping element with realistic parameters"""
    def __init__(self, timescale, x_tuple, rho_tuple, rho_init ):
        b = 1/timescale
        x_0 = (x_tuple[0] + x_tuple[1]) / 2
        a = - 4 * b / pow(x_tuple[1] - x_tuple[0], 2)
        
        self._rho_tuple = rho_tuple
        self._tip_width = 2 * np.sqrt(-4 * pow(b, 3) / (27 * a) )
        
        super().__init__( a = a, b = b, c = 0, x_0 = x_0)
        self._par['rho'] = rho_init
        self._update_par()
    
    def set_par(self, key, val):
        if key == 'c':
            raise ValueError("c is bounded to the control parameter rho!")
        self._par[key] = val
        self._update_par()
    
    def _update_par(self):
        rho1 = self._rho_tuple[0]
        rho2 = self._rho_tuple[1]
        width = self._tip_width
        rho = self._par['rho']
        self._par['c'] = width / (rho2 - rho1) * (rho - (rho1 + rho2) / 2)

class hopf(tipping_element):
    """hopf bifurcation in polar coordinates: dr/dt=(bif_par-r^2)*r*a, dphi/dt=b"""

    def __init__(self, a, c):
        super().__init__()
        self._type = 'hopf'
        self._par['a'] = a
        self._par['c'] = c

    def dxdt_diag(self):
        return lambda t, r: ( self._par['c'] - pow(r,2) ) * r * self._par['a']

    def jac_diag(self):
        return lambda t, r: self._par['a'] * self._par['c'] - \
                            self._par['a'] * 3 * pow(r,2)
    
    def tip_state(self):
        return lambda x: self._par['c'] > 0
