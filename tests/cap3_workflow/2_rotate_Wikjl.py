import numpy as np
from models.model_utils import dvr_to_bo

eigspec = np.load('eigspec.npy')
En = eigspec['En']
Cijn = eigspec['Cijn']

nbo = 1000
Wikjl = np.load('Wikjl.npy')
Wnm = dvr_to_bo(Cijn, Wikjl, nbo)
np.savetxt('Hnm.dat', np.diag(En[:nbo]))
np.savetxt('Wnm.dat', Wnm)


#print(np.load('eigspec.npy').shape)
#exit()
#En, Cijn = np.load('eigspec.npy')
#O = np.load('eigspec.npy')
#print(len(O))
#print(O[0].shape)
#print(O[1].shape)
#exit()

