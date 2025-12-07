import sys
import os
import numpy as np
from abc import ABC, abstractmethod

#sys.path.append(os.path.abspath("../src"))
from grid_utils.colbert_miller_dvr import dvr_xn, dvr_T, dvr_W

def dvr_to_bo(Cijn, Oikjl, nbo=None):
    ndvr = Cijn.shape[0]
    nfci = Cijn.shape[-1]
    CNn = Cijn.reshape(ndvr**2, nfci)[:,:nbo]
    ONM = Oikjl.reshape(ndvr**2, ndvr**2)
    Onm = np.linalg.multi_dot([CNn.T.conj(), ONM, CNn])
    return Onm

def calculate_nac(E, dHele_adi):
    nstates = E.shape[0]
    nac = np.zeros((nstates, nstates), dtype=np.complex128)
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                pass
            else:
                nac[i,j] = dHele_adi[i,j] / (E[j] - E[i])
    return nac

class Model(ABC):
    @abstractmethod
    def __init__(self, model_params):
        self.params = model_params

    @abstractmethod
    def VR(self, R):
        pass

    @abstractmethod
    def hij(self, R):
        pass

    #@abstractmethod
    #def map_ci(self):
    #    pass

    #@abstractmethod
    #def Hele(self, R):
    #    pass

    #@abstractmethod
    #def dHele(self, R):
    #    pass

    #@abstractmethod
    #def dipole(self, R):
    #    pass

    #@abstractmethod
    #def cap(self, R):
    #    pass


