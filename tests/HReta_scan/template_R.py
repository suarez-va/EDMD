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

model = TEGD(a=-25.0, b=25.0, N=75, bounds="(-inf,inf)", spin="triplet", model_params=params)

CIn = model.solve_wfn(R = R_sub, nbo = 250)

Wab20n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-20.0, bcap=20.0, ncap=2), CIn))
Wab40n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-40.0, bcap=40.0, ncap=2), CIn))
Wab50n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-50.0, bcap=50.0, ncap=2), CIn))
Wab60n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-60.0, bcap=60.0, ncap=2), CIn))
Wab80n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-80.0, bcap=80.0, ncap=2), CIn))
Wab100n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-100.0, bcap=100.0, ncap=2), CIn))
Wab120n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-120.0, bcap=120.0, ncap=2), CIn))
Wab150n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-150.0, bcap=150.0, ncap=2), CIn))

np.savez("boops", Wab20n2=Wab20n2, Wab40n2=Wab40n2, Wab50n2=Wab50n2, Wab60n2=Wab60n2, Wab80n2=Wab80n2, Wab100n2=Wab100n2, Wab120n2=Wab120n2, Wab150n2=Wab150n2)

