import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from grid_utils.colbert_miller_dvr import dvr_xn, dvr_T, dvr_W

from abc import ABC, abstractmethod

def dvr_to_bo(Cijn, Oikjl):
    ndvr = Cijn.shape[0]
    nbo = Cijn.shape[-1]
    CNn = Cijn.reshape(ndvr**2, nbo)
    ONM = Oikjl.reshape(ndvr**2, ndvr**2)
    Onm = np.linalg.multi_dot([CNn.conj().T, ONM, CNn])
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
    def __init__(self, a: float, b: float, N: int, bounds: str, spin: str, model_params: dict):
        self.a = a
        self.b = b
        self.N = N
        self.bounds = bounds
        self.spin = spin
        self.params = model_params
        assert self.bounds in ("(a,b)", "(0,inf)", "(-inf,inf)"), f"Colber Miller DVR Bounds: {self.bounds}, must be '(a,b)', '(0,inf)', or '(-inf,inf)'."
        assert self.spin in ("singlet", "triplet"), f"Spin Multiplicity: {self.spin}, must be 'singlet' or 'triplet'."

        self.ndvr = int(self.N-1) if self.bounds=="(a,b)" else int(self.N) if self.bounds=="(0,inf)" else int(self.N+1) if self.bounds=="(-inf,inf)" else 0
        self.nfci = int((self.ndvr+1)*self.ndvr/2) if self.spin=="singlet" else int(self.ndvr*(self.ndvr-1)/2) if self.spin=="triplet" else 0
        self.map_ij, self.map_kl = np.triu_indices(self.ndvr, k=0) if self.spin=="singlet" else np.triu_indices(self.ndvr, k=1) if self.spin=="triplet" else np.zeros((self.nfci), dtype=int)

        #self.ep = np.zeros((self.ndvr),dtype=np.complex128)
        #self.cip = np.zeros((self.ndvr,self.ndvr),dtype=np.complex128)
        #self.nbo = self.nfci if self.params["nbo"]==None else self.params["nbo"]
        #self.En = np.zeros((self.nbo), dtype=np.complex128)
        #self.Cijn = np.zeros((self.ndvr, self.ndvr, self.nbo), dtype=np.complex128)
 
    @abstractmethod
    def VR(self, R: float):
        pass

    @abstractmethod
    def dVR(self, R: float):
        pass

    @abstractmethod
    def VeR(self, x, R: float):
        pass

    @abstractmethod
    def dVeR(self, x, R: float):
        pass

    @abstractmethod
    def Vee(self, x1, x2):
        pass

    def xi(self) -> np.ndarray:
        return np.linspace(self.a, self.b, self.ndvr)

    # generate hcore using Colbert-Miller syle DVR
    def hij(self, R: float) -> np.ndarray:
        xi=self.xi()
        return dvr_T(1, self.a, self.b, self.N, self.bounds) + np.diag(self.VeR(xi, R))

    # generate derivative of hcore with respect to R using Colbert-Miller syle DVR for kinetic energy
    def dhij(self, R: float) -> np.ndarray:
        xi=self.xi()
        return np.diag(self.dVeR(xi, R))

    def solve_mos(self, R: float):
        ep, cip = eigh(self.hij(R))
        np.savez('mospec', xi=self.xi(), ep=ep, cip=cip)
        return ep, cip

    def Vik(self) -> np.ndarray:
        xi=self.xi()
        return self.Vee(xi[:,None],xi[None,:])

    def Hikjl(self, R: float) -> np.ndarray:
        Hikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        hij = self.hij(R)
        Vik = self.Vik()
        Hikjl += hij[:,None,:,None] * dij[None,:,None,:]
        Hikjl += dij[:,None,:,None] * hij[None,:,None,:]
        Hikjl += Vik[:,:,None,None] * dij[:,None,:,None] * dij[None,:,None,:]
        return Hikjl

    def dHikjl(self, R: float) -> np.ndarray:
        dHikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        dhij = self.dhij(R)
        dHikjl += dhij[:,None,:,None] * dij[None,:,None,:]
        dHikjl += dij[:,None,:,None] * dhij[None,:,None,:]
        return dHikjl

    def Hdvr(self, R: float) -> np.ndarray:
        H = np.zeros((self.nfci, self.nfci), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        Hikjl = self.Hikjl(R)
        match self.spin:
            case "singlet":
                Hikjl *= (1 - (1 - 1 / np.sqrt(2)) * dij[:,:,None,None]) * (1 - (1 - 1 / np.sqrt(2)) * dij[None,None,:,:])
                H += Hikjl[self.map_ij[:,None], self.map_kl[:,None], self.map_ij[None,:], self.map_kl[None,:]]
                H += Hikjl[self.map_ij[:,None], self.map_kl[:,None], self.map_kl[None,:], self.map_ij[None,:]]
            case "triplet":
                H += Hikjl[self.map_ij[:,None], self.map_kl[:,None], self.map_ij[None,:], self.map_kl[None,:]]
                H -= Hikjl[self.map_ij[:,None], self.map_kl[:,None], self.map_kl[None,:], self.map_ij[None,:]]
        return H

    def solve_wfn(self, R: float, nbo: int = 0):
        if nbo == 0 or nbo == self.nfci:
             Cijn = np.zeros((self.ndvr, self.ndvr, self.nfci), dtype=np.complex128)
             En, Cijn[self.map_ij, self.map_kl, :] = eigh(self.Hdvr(R))
        else:
             Cijn = np.zeros((self.ndvr, self.ndvr, nbo), dtype=np.complex128)
             En, Cijn[self.map_ij, self.map_kl, :] = eigsh(self.Hdvr(R), k=nbo, which='SA')
        match self.spin:
            case "singlet":
                Cijn += Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
                Cijn[np.arange(self.ndvr),np.arange(self.ndvr),:] *= 1 / np.sqrt(2)
            case "triplet":
                Cijn += -Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
        np.savez('eigspec', xi=self.xi(), En=En, Cijn=Cijn)
        return En, Cijn

    def xikjl(self, n: int = 1) -> np.ndarray:
        xikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        xij = dvr_xn(n, self.a, self.b, self.N, self.bounds)
        xikjl += xij[:,None,:,None] * dij[None,:,None,:]
        xikjl += dij[:,None,:,None] * xij[None,:,None,:]
        return xikjl

    def Wikjl(self, acap: float, bcap: float, ncap: int = 2) -> np.ndarray:
        Wikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        wij = dvr_W(self.a, self.b, self.N, acap, bcap, ncap, self.bounds)
        Wikjl += wij[:,None,:,None] * dij[None,:,None,:]
        Wikjl += dij[:,None,:,None] * wij[None,:,None,:]
        return Wikjl


#class Model(ABC):
#    @abstractmethod
#    def __init__(self, model_params):
#        self.params = model_params
#
#    @abstractmethod
#    def VR(self, R):
#        pass
#
#    @abstractmethod
#    def hij(self, R):
#        pass
#
#    #@abstractmethod
#    #def map_ci(self):
#    #    pass
#
#    #@abstractmethod
#    #def Hele(self, R):
#    #    pass
#
#    #@abstractmethod
#    #def dHele(self, R):
#    #    pass
#
#    #@abstractmethod
#    #def dipole(self, R):
#    #    pass
#
#    #@abstractmethod
#    #def cap(self, R):
#    #    pass


