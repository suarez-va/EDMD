import numpy as np
from model_systems.models import GICD
from time_independent.fcidvr import FCIDVR, compute_density, compute_dyson, compute_cap

params = {
    'ZA': 0.0,
    'ZB': 0.0,
    'aR': 0.0,
    'bR': 0.0001,
    'DA' : 1.0,
    'bA': 0.25,
    'DB' : 0.8,
    'bB': 1.0,
    'aee': 0.0,
    'bee': 0.0001,
}

R = 8.0

L = 266.00000000000
xN = int(4*L)

nbo1 = 25
nbo2 = 250

model = GICD(params)
fcidvr = FCIDVR(model = model, xa = -L/2.0, xb = L/2.0, xN = xN, xbounds = "(-inf,inf)")

fcidvr.kernel(R = R, nele = 1, spin = 'doublet', nbo = nbo1, derivative_order = 0)
fcidvr.kernel(R = R, nele = 2, spin = 'singlet', nbo = nbo2, derivative_order = 0)

#compute_density(fcidvr_file = 'fcidvr_1ele_doublet.npz', Sz = 0.5)
#compute_density(fcidvr_file = 'fcidvr_1ele_doublet.npz', Sz = -0.5)
#compute_density(fcidvr_file = 'fcidvr_2ele_singlet.npz', Sz = 0.0)

#compute_dyson('fcidvr_1ele_doublet.npz', 0.5, 'fcidvr_2ele_singlet.npz', 0.0)
#compute_dyson('fcidvr_1ele_doublet.npz', -0.5, 'fcidvr_2ele_singlet.npz', 0.0)

#compute_cap(fcidvr_file = 'fcidvr_1ele_doublet.npz', acap = acap, bcap = bcap, ncap = ncap)
#compute_cap(fcidvr_file = 'fcidvr_2ele_singlet.npz', acap = acap, bcap = bcap, ncap = ncap)

