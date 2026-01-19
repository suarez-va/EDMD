import numpy as np
from models.model_utils import dvr_to_bo
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

model = TEGD(a=-150.0, b=150.0, N=325, bounds="(-inf,inf)", spin="triplet", model_params=params)

En, Cijn = model.solve_wfn(R = R_sub, nbo = 100)

Wab20n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-20.0, bcap=20.0, ncap=2))
Wab40n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-40.0, bcap=40.0, ncap=2))
Wab50n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-50.0, bcap=50.0, ncap=2))
Wab60n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-60.0, bcap=60.0, ncap=2))
Wab80n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-80.0, bcap=80.0, ncap=2))
np.savez("boops", Wab20n2=Wab20n2, Wab40n2=Wab40n2, Wab50n2=Wab50n2, Wab60n2=Wab60n2, Wab80n2=Wab80n2)

