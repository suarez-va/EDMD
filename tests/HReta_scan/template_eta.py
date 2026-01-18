import numpy as np
from models.model_utils import solve_cap

eta = eta_sub

eigsolve = np.load('../../eigspec2.npz')
boops = np.load('../../boops2.npz')
Hnm = np.diag(eigsolve['En'])
#Wnm = boops["Wab15n2"]
Wnm = boops["Wab20n2"]
En, Cnml, Cnmr = solve_cap(Hnm, Wnm, eta)

