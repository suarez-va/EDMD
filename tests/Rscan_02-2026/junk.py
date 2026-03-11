import numpy as np

ndvr = 5

one = np.zeros((ndvr,2), dtype=np.complex128)
two = np.zeros((ndvr,2,ndvr,2), dtype=np.complex128)

print(one.shape)
print(one[:,0].shape)

print(two.shape)
print(two[:,0,:,1].shape)

allowed_cases = {
    1: [[0.5, 0.5], [0.5, -0.5]],
    2: [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, -1.0]],
    3: [[0.5, 0.5], [0.5, -0.5]]
}

#print(allowed_cases[2][0])

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
fcidvr = FCIDVR(model = model, xa = -196.7/2.0, xb = 196.7/2.0, xN = 250, xbounds = "(-inf,inf)")
fcidvr.validate_nele_spin(nele = 2, S = 1.0, Sz = 0.5)
