import numpy as np
from scipy.linalg import eig
import time

eta = REPLACE

En, Cijn = np.load('../../eigenspectra.npy')
Wnm = np.loadtxt('../../Wnm.dat', dtype=np.complex128)
nbo = Wnm.shape[0]
H = np.diag(En[:nbo]) - 1j * eta *  Wnm
t1 = time.time()
E, C = eig(H)
t2 = time.time()
print(f'eig(): t={t2-t1}')

Er = E[:].real
Gam = E[:].imag

np.savetxt('Er.dat', Er)
np.savetxt('Gam.dat', Gam)
np.save('Cnm', C)

