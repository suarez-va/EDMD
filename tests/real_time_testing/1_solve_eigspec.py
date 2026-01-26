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

#model = TEGD(a=-196.7/2.0, b=196.7/2.0, N=500, bounds="(-inf,inf)", spin="singlet", model_params=params)
model = TEGD(a=-196.7/2.0, b=196.7/2.0, N=250, bounds="(-inf,inf)", spin="singlet", model_params=params)

ep, cip = model.solve_mos(R = 8.0)
wab50n2 = model.wij(acap=-50.0, bcap=50.0, ncap=2)
np.savez("moops", wab50n2=wab50n2)

CIn = model.solve_wfn(R = 8.0, nbo = 225)
Wab50n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-50.0, bcap=50.0, ncap=2), CIn))
np.savez("boops", Wab50n2=Wab50n2)

exit()

ep, cip = model.solve_mos(R = 8.0)

