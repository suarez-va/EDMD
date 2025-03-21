import numpy as np
from models.two_electron_screened_diatomic import generate_Hele, generate_dHele, calculate_nac, generate_dipole, generate_cap

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

Hele = generate_Hele(19.80000000000, params)
E, C = np.linalg.eigh(Hele)

dHele = np.linalg.multi_dot([np.conj(C.T), generate_dHele(19.80000000000, params), C])
nac1 = calculate_nac(E, dHele)
nac2 = np.einsum('ij,jk->ik', nac1, nac1)
dipole = np.linalg.multi_dot([np.conj(C.T), generate_dipole(19.80000000000, params), C])
cap = np.linalg.multi_dot([np.conj(C.T), generate_cap(19.80000000000, params), C])

np.savetxt('E.dat', E)
np.savetxt('C.dat', C)
np.savetxt('dHele.dat', dHele)
np.savetxt('nac1.dat', nac1)
np.savetxt('nac2.dat', nac2)
np.savetxt('dipole.dat', dipole)
np.savetxt('cap.dat', cap)

