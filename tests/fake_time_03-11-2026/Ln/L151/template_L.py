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

L = 251.00000000000
xN = int(2*L)

nbo1 = 25
nbo2 = 250

model = GICD(params)
fcidvr = FCIDVR(model = model, xa = -L/2.0, xb = L/2.0, xN = xN, xbounds = "(-inf,inf)")

fcidvr.kernel(R = R, nele = 1, spin = 'doublet', nbo = nbo1, derivative_order = 0)
fcidvr.kernel(R = R, nele = 2, spin = 'singlet', nbo = nbo2, derivative_order = 0)
