import numpy as np
from grid_utils.colbert_miller_dvr import dvr_p, dvr_T, dvr_xn, dvr_W
from models.two_electron_screened_diatomic import TESD, generate_Hele, generate_dHele, generate_dipole, generate_cap, Vnuc
import time

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
    "ndvr": 250,
    "ncas": 250,
    "xcap": 12.5,
    "eta": 0,
    "ncap": 4,
    "spin": "singlet"
}

R = 2.5
model = TESD(params)
t1 = time.time()
model.solve_wfn(R)
t2 = time.time()
En = model.En
Cijn = model.Cijn
np.save('eigenspectra', En, Cijn)
print(f'solve_wfn(): t={t2-t1}')
Wnm = model.Wnm(nbo=250)
t3 = time.time()
print(f'Wnm(): t={t3-t2}')
np.savetxt('Wnm.dat', Wnm)

