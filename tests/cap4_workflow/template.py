import numpy as np
from scipy.linalg import eig

eta = REPLACE

Hnm = np.loadtxt('../../Hnm.dat', dtype=np.complex128)
Wnm = np.loadtxt('../../Wnm.dat', dtype=np.complex128)
Hcap = Hnm - 1j * eta *  Wnm
E, Cl, Cr = eig(Hcap, left=True, right=True)
idx = np.argsort(E.real)
E, Cl, Cr = E[idx], Cl[:,idx], Cr[:,idx]
alpha = np.sqrt(np.diag(np.matmul(Cl.conj().T, Cr)))
Cl *= 1 / alpha
Cr *= 1 / alpha.conj()

#print(np.matmul(Cl.conj().T , Cr))

np.savetxt('Er.dat', E.real)
np.savetxt('Gam.dat', E.imag)
np.savetxt('Clnm.dat', Cl)
np.savetxt('Crnm.dat', Cr)

