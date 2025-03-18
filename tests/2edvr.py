import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))
from colbert_miller_dvr import dvr_p, dvr_T
from models.two_electron_screened_diatomic import generate_Hele

R = 2.08
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
    "xmax": 25.0,
    "xpts": 25
}

print(generate_Hele(R, params))

