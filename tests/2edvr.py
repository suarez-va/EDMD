import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))
from colbert_miller_dvr import dvr_p, dvr_T
from models.two_electron_screened_diatomic import generate_Hele, generate_dipole, generate_cap

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
    "xpts": 25,
    "xcap": 17.5,
    "etacap": 1.0,
    "ncap": 3
}

print("1:")
print(generate_Hele(R, params))

print("2:")
print(generate_dipole(R, params))

print("3:")
print(generate_cap(R, params))

