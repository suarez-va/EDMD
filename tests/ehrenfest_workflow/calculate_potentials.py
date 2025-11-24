import numpy as np
from models.two_electron_screened_diatomic import generate_Hele, generate_dHele, generate_dipole, generate_cap, Vnuc

class model_system:
    def __init__(self, model_params):
        self.model_params = model_params
    def Hele(self, R):
        return generate_Hele(R, self.model_params)
    def dHele(self, R):
        return generate_dHele(R, self.model_params)
    def Wext(self, R, t):
        return generate_dipole(R, self.model_params)

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
    "xmax": 10.0,
    "xpts": 125,
    "xcap": 17.5,
    "etacap": 1.0,
    "ncap": 3
}

Rpts=7
nsta=12
R=np.linspace(1.0, 4.0, Rpts)
#E=np.zeros((Rpts,nsta), dtype=np.complex128)
ER=np.zeros((Rpts,nsta))
Gamma=np.zeros((Rpts,nsta))
for i in range(Rpts):
    print(i)
    Ri = R[i]
    Hele = generate_Hele(Ri, params)
    #Ei, Ci = np.linalg.eig(Hele)
    Ei, Ci = np.linalg.eigh(Hele)
    ER[i,:]=Ei[:nsta] + Vnuc(Ri, params)
    #ER[i,:]=Ei[:nsta].real
    #Gamma[i,:]=Ei[:nsta].imag

np.savetxt('R.dat', R)
np.savetxt('ER.dat', ER)
np.savetxt('Gamma.dat', Gamma)
