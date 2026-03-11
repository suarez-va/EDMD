from model_systems.models import GICD, BHAR
from time_independent.fcidvr import FCIDVR

params = {
    'ZA': 0.5,
    'ZB': 0.5,
    'aR': 0.0,
    'bR': 0.0001,
    'DA' : 1.0,
    'bA': 0.25,
    'DB' : 0.8,
    'bB': 1.0,
    'aee': 0.0,
    'bee': 0.0001,
}

params2 = {
    'ZA': 1.0,
    'ZB': 1.0,
    'aR': 0.0,
    'bR': 0.0001,
    'k': 0.05,
    'mu_mA': 0.9,
    'aee': 0.0,
    'bee': 0.0001,
}

model = GICD(params)
#model = BHAR(params2)
#fcidvr = FCIDVR(model = model, xa = -196.7/2.0, xb = 196.7/2.0, xN = 250, xbounds = "(-inf,inf)")
#fcidvr.kernel(R = 3.20000000000, nele = 1, spin = 'doublet', nbo = 125, derivative_order=1)
#fcidvr.kernel(R = 3.20000000000, nele = 2, spin = 'singlet', nbo = 50, derivative_order=1)

#fcidvr = FCIDVR(model = model, xa = -196.7/20.0, xb = 196.7/20.0, xN = 25, xbounds = "(-inf,inf)")
#fcidvr.kernel(R = 3.20000000000, nele = 1, spin = 'doublet', nbo = 26, derivative_order=1)
#fcidvr.kernel(R = 3.20000000000, nele = 2, spin = 'singlet', nbo = 26, derivative_order=1)
#fcidvr.kernel(R = 3.20000000000, nele = 2, spin = 'singlet', nbo = 324, derivative_order=1)

fcidvr = FCIDVR(model = model, xa = -196.7/2.0, xb = 196.7/2.0, xN = 250, xbounds = "(-inf,inf)")
fcidvr.kernel(R = 3.20000000000, nele = 1, spin = 'doublet', nbo = 251, derivative_order=1)
#fcidvr.kernel(R = 3.20000000000, nele = 2, spin = 'singlet', nbo = 49, derivative_order=1)
#fcidvr.kernel(R = 3.20000000000, nele = 2, spin = 'singlet', nbo = 1274)

#fcidvr = FCIDVR(model = model, xa = -25, xb = 25, xN = 100, xbounds = "(-inf,inf)")
#fcidvr.kernel(R = 3.20000000000, nele = 1, spin = 'doublet', nbo = 101, derivative_order=2)

