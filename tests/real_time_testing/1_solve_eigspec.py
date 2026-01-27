import numpy as np
from models.model_utils import matmat
from models.two_electron_diatomic import TEGD

params = {
    "aR": 0.0,
    "bR": 0.0001,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.0,
    "bee": 0.0001,
}

model = TEGD(a=-196.7/2.0, b=196.7/2.0, N=750, bounds="(-inf,inf)", spin="singlet", model_params=params)

R = 8.0
acap = -50.0
bcap = 50.0
ncap = 2

Cn = model.solve_bo(R = R, nbo = 225)
Wnm = np.matmul(Cn.conj().T, matmat(model.W(acap = acap, bcap = bcap, ncap = ncap), Cn))
np.savez("boops", Wnm=Wnm)

cip = model.solve_mos(R = R)
wij = model.wij(acap = acap, bcap = bcap, ncap = ncap)
wk = np.diag(wij)
wpq = np.matmul(cip.conj().T, np.matmul(wij, cip))
np.savez("moops", wk = wk, wpq = wpq)
