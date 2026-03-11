from model_systems.models import GICD
from time_independent.fcidvr import validate_nele_spin, fci_mapping, fci_operator, fci_wfn, FCIDVR, compute_cap
import numpy as np

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


R_sub = 8.0
nbo_sub = 75

model = GICD(params)
fcidvr = FCIDVR(model = model, xa = -196.7/2.0, xb = 196.7/2.0, xN = 250, xbounds = "(-inf,inf)")

ndvr = fcidvr.nxdvr
nfci_singlet, map_norm_singlet, map_idx_singlet = fci_mapping(ndvr = ndvr, nele = 2, spin = 'singlet')

fcidvr.kernel(R = R_sub, nele = 2, spin = 'singlet', nbo = nbo_sub, derivative_order = 0)

compute_cap(fcidvr_file = 'fcidvr_2ele_singlet.npz', acap = -50.0, bcap = 50.0, ncap = 2)

