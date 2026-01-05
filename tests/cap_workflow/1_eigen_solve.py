import numpy as np
from models.two_electron_gaussian_diatomic import TEGD
from models.model_utils import dvr_to_bo

params = {
    "aR": 0.01205,
    "bR": 0.01,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.1,
    "bee": 100.0,
}

R = 8.5
ab = 40.0
N = 120

model = TEGD(a=-ab, b=ab, N=N, bounds="(-inf,inf)", spin="triplet",model_params=params)
En, Cijn = model.solve_wfn(R)

acap = -25.0
bcap = 25.0
ncap = 4
Wikjl = model.Wikjl(acap, bcap, ncap)
Wnm = dvr_to_bo(Cijn[:,:,:500], Wikjl)
np.savetxt('Hnm.dat', np.diag(En[:500]))
np.savetxt('Wnm.dat', Wnm)

