import numpy as np
from models.model_utils import dvr_to_bo

eigspec = np.load('eigspec.npz')
En = eigspec['En']
xi = eigspec['xi']
Cijn = eigspec['Cijn']

nbo = 250
Wikjl = np.load('Wikjl.npy')
Wnm = dvr_to_bo(Cijn, Wikjl, nbo)
np.savetxt('Hnm.dat', np.diag(En[:nbo]))
np.savetxt('Wnm.dat', Wnm)


