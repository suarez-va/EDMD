import numpy as np
from models.two_electron_screened_diatomic import generate_Hele, generate_dipole, generate_cap

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

Hele = generate_Hele(REPLACE, params)
dipole = generate_dipole(REPLACE, params)
cap = generate_cap(REPLACE, params)

E, C = np.linalg.eigh(Hele)

np.savetxt('E.dat', E)
np.savetxt('C.dat', C)
np.savetxt( 'dipole.dat', np.linalg.multi_dot([np.conj(C.T), dipole, C]))
np.savetxt( 'cap.dat', np.linalg.multi_dot([np.conj(C.T), cap, C]))

