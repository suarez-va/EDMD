import numpy as np
from models.model_utils import solve_cap

eta = eta_sub

eigsolve = np.load('../../eigspec.npz')
boops = np.load('../../boops.npz')
Hnm = np.diag(eigsolve['En'])
#Wnm = boops["Wab20n2"]
#Wnm = boops["Wab40n2"]
Wnm = boops["Wab50n2"]

En, Cnml, Cnmr = solve_cap(Hnm, Wnm, eta)

