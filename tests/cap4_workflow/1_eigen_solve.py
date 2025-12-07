import numpy as np
from models.two_electron_screened_diatomic import TESD

params = {
    "aee": 0.02,
    "bee": 0.25,
    "aR": 0.01205,
    "bR": 0.01,
    "aAe": 0.0102,
    "bAe": 0.655,
    "aBe": 0.0139,
    "bBe": 0.473,
    "mA": 36443.98900696,
    "mB": 7294.29954142,
    "xmax": 40.0,
    "ndvr": 250,
    "spin": "singlet",
    "nbo": 250
}

R = 2.5
model = TESD(params)
model.solve_wfn(R)

acap = -25.0
bcap = 25.0
ncap = 4
Wikjl = model.Wikjl(acap, bcap, ncap)
np.save('Wikjl', Wikjl)

