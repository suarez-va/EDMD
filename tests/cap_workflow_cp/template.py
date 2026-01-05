import numpy as np
from grid_utils.colbert_miller_dvr import dvr_p, dvr_T, dvr_xn, dvr_W
from models.two_electron_screened_diatomic import TESD, generate_Hele, generate_dHele, generate_dipole, generate_cap, Vnuc

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
    # max=20 then ndvr=150, but do more honestly if you want many digits
    "xmax": 30.0,
    "ndvr": 300,
    "ncas": 300,
    "xcap": 20.0,
    "eta": REPLACE,
    "ncap": 4,
    "spin": "singlet"
}

R = 2.5
model = TESD(params)
model.solve_wfn(R)
E = model.En
Cijn = model.Cijn

Er = E[:].real
Gam = E[:].imag

np.savetxt('Er.dat', Er)
np.savetxt('Gam.dat', Gam)
np.save('Cijn', Cijn)

