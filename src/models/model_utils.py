import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigsh
from grids.colbert_miller_dvr import dvr_x, dvr_T, dvr_W

from abc import ABC, abstractmethod

def dvr_to_bo(Cijn, Oikjl):
    ndvr = Cijn.shape[0]
    nbo = Cijn.shape[-1]
    CNn = Cijn.reshape(ndvr**2, nbo)
    ONM = Oikjl.reshape(ndvr**2, ndvr**2)
    Onm = np.linalg.multi_dot([CNn.conj().T, ONM, CNn])
    return Onm

def solve_nac1(En, d1Hnm):
    nbo = En.shape[0]
    nac1 = np.zeros((nbo, nbo), dtype=np.complex128)
    mask = np.ones((nbo, nbo), dtype=bool)
    np.fill_diagonal(mask, False)
    dE = En[:, None] - En[None, :]
    nac1[mask] = -d1Hnm[mask] / dE[mask]
    return nac1

def solve_d2En(En, d1Hnm, d2Hnm):
    nbo = En.shape[0]
    d2En = np.zeros((nbo), dtype=np.float64)
    M = np.zeros((nbo, nbo), dtype=np.float64)
    mask = np.ones((nbo, nbo), dtype=bool)
    np.fill_diagonal(mask, False)
    dE = En[:, None] - En[None, :]
    M[mask] = np.absolute(d1Hnm[mask])**2 / dE[mask]
    d2En = np.diagonal(d2Hnm.real) + 2 * np.sum(M, axis=1)
    return d2En

def solve_cap(Hnm, Wnm, eta):
    Hcap = Hnm - 1j * eta *  Wnm
    En, Cnml, Cnmr = eig(Hcap, left=True, right=True)
    idx = np.argsort(En.real)
    En, Cnml, Cnmr = En[idx], Cnml[:,idx], Cnmr[:,idx]
    norm = np.sqrt(np.diag(np.matmul(Cnml.conj().T, Cnmr)))
    Cnml *= 1 / norm; Cnmr *= 1 / norm.conj()
    np.savez('capspec', En=En, Cnml=Cnml, Cnmr=Cnmr)
    return En, Cnml, Cnmr

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

    @abstractmethod
    def VR(self, R: float):
        pass

    @abstractmethod
    def d1VR(self, R: float):
        pass

    @abstractmethod
    def d2VR(self, R: float):
        pass

    @abstractmethod
    def VeR(self, x, R: float):
        pass

    @abstractmethod
    def d1VeR(self, x, R: float):
        pass

    @abstractmethod
    def d2VeR(self, x, R: float):
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
    def d1hij(self, R: float) -> np.ndarray:
        xi=self.xi()
        return np.diag(self.d1VeR(xi, R))

    def d2hij(self, R: float) -> np.ndarray:
        xi=self.xi()
        return np.diag(self.d2VeR(xi, R))

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

    def d1Hikjl(self, R: float) -> np.ndarray:
        d1Hikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        d1hij = self.d1hij(R)
        d1Hikjl += d1hij[:,None,:,None] * dij[None,:,None,:]
        d1Hikjl += dij[:,None,:,None] * d1hij[None,:,None,:]
        return d1Hikjl

    def d2Hikjl(self, R: float) -> np.ndarray:
        d2Hikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        d2hij = self.d1hij(R)
        d2Hikjl += d2hij[:,None,:,None] * dij[None,:,None,:]
        d2Hikjl += dij[:,None,:,None] * d2hij[None,:,None,:]
        return d2Hikjl

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
             idx = np.argsort(En)
             En = En[idx]; Cijn = Cijn[:,:,idx]
        match self.spin:
            case "singlet":
                Cijn += Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
                Cijn[np.arange(self.ndvr),np.arange(self.ndvr),:] *= 1 / np.sqrt(2)
            case "triplet":
                Cijn += -Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
        d1Hnm = dvr_to_bo(Cijn, self.d1Hikjl(R))
        d1En = np.diag(d1Hnm.real)
        nac1 = solve_nac1(En, d1Hnm)
        d2Hnm = dvr_to_bo(Cijn, self.d1Hikjl(R))
        d2En = solve_d2En(En, d1Hnm, d2Hnm)
        np.savez('eigspec', xi=self.xi(), En=En, Cijn=Cijn, d1En=d1En, d1Hnm=d1Hnm, nac1=nac1, d2En=d2En, d2Hnm=d2Hnm)
        return En, Cijn

    def solve_wfn_oldish(self, R: float, nbo: int = 0, grad: int = 2):
        if nbo == 0 or nbo == self.nfci:
             Cijn = np.zeros((self.ndvr, self.ndvr, self.nfci), dtype=np.complex128)
             En, Cijn[self.map_ij, self.map_kl, :] = eigh(self.Hdvr(R))
        else:
             Cijn = np.zeros((self.ndvr, self.ndvr, nbo), dtype=np.complex128)
             En, Cijn[self.map_ij, self.map_kl, :] = eigsh(self.Hdvr(R), k=nbo, which='SA')
             idx = np.argsort(En)
             En = En[idx]; Cijn = Cijn[:,:,idx]
        match self.spin:
            case "singlet":
                Cijn += Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
                Cijn[np.arange(self.ndvr),np.arange(self.ndvr),:] *= 1 / np.sqrt(2)
            case "triplet":
                Cijn += -Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
        if grad == 0:
            np.savez('eigspec', xi=self.xi(), En=En, Cijn=Cijn)
        elif grad == 1:
            d1Hnm = dvr_to_bo(Cijn, self.d1Hikjl(R))
            d1En = np.diag(d1Hnm.real)
            nac1 = solve_nac1(En, d1Hnm)
            np.savez('eigspec', xi=self.xi(), En=En, Cijn=Cijn, d1En=d1En, d1Hnm=d1Hnm, nac1=nac1)
        elif grad == 2:
            d1Hnm = dvr_to_bo(Cijn, self.d1Hikjl(R))
            d1En = np.diag(d1Hnm.real)
            nac1 = solve_nac1(En, d1Hnm)
            d2Hnm = dvr_to_bo(Cijn, self.d1Hikjl(R))
            d2En = solve_d2En(En, d1Hnm, d2Hnm)
            np.savez('eigspec', xi=self.xi(), En=En, Cijn=Cijn, d1En=d1En, d1Hnm=d1Hnm, nac1=nac1, d2En=d2En, d2Hnm=d2Hnm)
        else:
            print("grad should be either 0, 1, or 2")
        return En, Cijn



    def solve_wfn_old(self, R: float, nbo: int = 0):
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

    def xikjl(self) -> np.ndarray:
        xikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        xij = dvr_x(self.a, self.b, self.N, self.bounds)
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

