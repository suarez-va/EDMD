import numpy as np
from scipy.linalg import eigh, eig
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import eigsh, LinearOperator
from grids.colbert_miller_dvr import dvr_x, dvr_T, dvr_W

from abc import ABC, abstractmethod

def matmat(operator, vectors):
    k = vectors.shape[1]
    out = np.zeros_like(vectors, dtype=np.complex128)
    for j in range(k):
        out[:, j] = operator.matvec(vectors[:, j])
    return out

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
    #Cnml *= 1 / norm; Cnmr *= 1 / norm.conj() YOU HAD THIS ONE FLIPPED FOR SOME TIME!!!
    Cnml *= 1 / norm.conj(); Cnmr *= 1 / norm
    np.savez('capspec', En=En, Cnml=Cnml, Cnmr=Cnmr)
    return En, Cnml, Cnmr

def dvr_to_bo(Cijn, Oikjl):
    ndvr = Cijn.shape[0]
    nbo = Cijn.shape[-1]
    CNn = Cijn.reshape(ndvr**2, nbo)
    ONM = Oikjl.reshape(ndvr**2, ndvr**2)
    Onm = np.linalg.multi_dot([CNn.conj().T, ONM, CNn])
    return Onm

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

    def Vik(self) -> np.ndarray:
        xi=self.xi()
        return self.Vee(xi[:,None],xi[None,:])

    def solve_mos(self, R: float):
        hij = self.hij(R)
        ep, cip = eigh(hij)

        hpq = np.matmul(cip.conj().T, np.matmul(hij, cip))
        np.savez('mospec', xi=self.xi(), ep=ep, cip=cip, hpq=hpq)
        return cip

    def wfn_to_vec(self, Cijn):
        assert (Cijn.shape[0] == self.ndvr) and (Cijn.shape[1] == self.ndvr), f"First two dimensions of wavefunction Cijn, must match dvr dimensionality: ({self.ndvr}, {self.ndvr})"
        assert (Cijn.ndim == 2) or (Cijn.ndim == 3), f"Wavefunction Cijn should have either 2 or 3 dimensions"
        if Cijn.ndim==2:
            nbo = 0
            Cijn = Cijn[:,:,None]
            CIn = np.zeros((self.nfci, 1), dtype=np.complex128)
        else:
            nbo = Cijn.shape[2]
            CIn = np.zeros((self.nfci, nbo), dtype=np.complex128)
        match self.spin:
            case "singlet":
                dij = np.eye(self.ndvr)
                nij = (1 - (1 - 1 / np.sqrt(2)) * dij)
                CIn = nij[self.map_ij, self.map_kl, None] / np.sqrt(2) * (Cijn[self.map_ij, self.map_kl, :] + Cijn[self.map_kl, self.map_ij, :])
            case "triplet":
                CIn = 1 / np.sqrt(2) * (Cijn[self.map_ij, self.map_kl, :] - Cijn[self.map_kl, self.map_ij, :])
        return CIn[:, 0] if (nbo==0) else CIn

    def vec_to_wfn(self, CIn):
        assert (CIn.shape[0] == self.nfci), f"First dimension of CI vector CIn, must match fci dimensionality: ({self.nfci})"
        assert (CIn.ndim == 1) or (CIn.ndim == 2), f"CI vector CIn should have either 1 or 2 dimensions"
        if CIn.ndim==1:
            nbo = 0
            CIn = CIn[:,None]
            Cijn = np.zeros((self.ndvr, self.ndvr, 1), dtype=np.complex128)
        else:
            nbo = CIn.shape[1]
            Cijn = np.zeros((self.ndvr, self.ndvr, nbo), dtype=np.complex128)
        Cijn[self.map_ij, self.map_kl, :] = CIn
        match self.spin:
            case "singlet":
                dij = np.eye(self.ndvr)
                nij = (1 - (1 - 1 / np.sqrt(2)) * dij)
                Cijn *= nij[:,:,None]
                Cijn += Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
            case "triplet":
                Cijn += -Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
        return Cijn[:, :, 0] if (nbo==0) else Cijn

    def H(self, R: float):
        dij = np.eye(self.ndvr)
        nij = (1 - (1 - 1 / np.sqrt(2)) * dij)
        hij = self.hij(R)
        Vik = self.Vik()
        match self.spin:
            case "singlet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl *= nij; Cjl += Cjl.T
                    hC = np.matmul(hij, Cjl)
                    Cik = nij * (hC + hC.T + Vik * Cjl)
                    #Cik = nij * (np.matmul(hij, Cjl) + np.matmul(Cjl, hij.conj().T) + Vik * Cjl)
                    return Cik[self.map_ij, self.map_kl]
            case "triplet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl -= Cjl.T
                    Cik = (np.matmul(hij, Cjl) + np.matmul(Cjl, hij.conj().T) + Vik * Cjl)
                    return Cik[self.map_ij, self.map_kl]
        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, dtype=np.complex128)

    def d1H(self, R: float):
        dij = np.eye(self.ndvr)
        nij = (1 - (1 - 1 / np.sqrt(2)) * dij)
        d1hij = self.d1hij(R)
        match self.spin:
            case "singlet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl *= nij; Cjl += Cjl.T
                    Cik = nij * (np.matmul(d1hij, Cjl) + np.matmul(Cjl, d1hij.conj().T))
                    return Cik[self.map_ij, self.map_kl]
            case "triplet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl -= Cjl.T
                    Cik = (np.matmul(d1hij, Cjl) + np.matmul(Cjl, d1hij.conj().T))
                    return Cik[self.map_ij, self.map_kl]
        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, dtype=np.complex128)

    def d2H(self, R: float):
        dij = np.eye(self.ndvr)
        nij = (1 - (1 - 1 / np.sqrt(2)) * dij)
        d2hij = self.d2hij(R)
        match self.spin:
            case "singlet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl *= nij; Cjl += Cjl.T
                    Cik = nij * (np.matmul(d2hij, Cjl) + np.matmul(Cjl, d2hij.conj().T))
                    return Cik[self.map_ij, self.map_kl]
            case "triplet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl -= Cjl.T
                    Cik = (np.matmul(d2hij, Cjl) + np.matmul(Cjl, d2hij.conj().T))
                    return Cik[self.map_ij, self.map_kl]
        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, dtype=np.complex128)

    def solve_bo(self, R: float, nbo: int = 100):
        En, Cn = eigsh(self.H(R), k=nbo, which='SA')
        #En, Cn = eigsh(self.H(R), k=nbo, which='SA', ncv=~1.5*k, tol=1e-8)
        idx = np.argsort(En)
        En = En[idx]; Cn = Cn[:,idx]

        d1Hnm = np.matmul(Cn.conj().T, matmat(self.d1H(R), Cn))
        d1En = np.diag(d1Hnm.real)
        nac1 = solve_nac1(En, d1Hnm)
        d2Hnm = np.matmul(Cn.conj().T, matmat(self.d2H(R), Cn))
        d2En = solve_d2En(En, d1Hnm, d2Hnm)
        Cijn = self.vec_to_wfn(Cn)
        np.savez('bospec', xi=self.xi(), En=En, Cijn=Cijn, d1En=d1En, d1Hnm=d1Hnm, nac1=nac1, d2En=d2En, d2Hnm=d2Hnm)
        return Cn

    # generate cap using Colbert-Miller syle DVR
    def wij(self, acap: float, bcap: float, ncap: int = 2) -> np.ndarray:
        return dvr_W(self.a, self.b, self.N, acap, bcap, ncap, self.bounds)

    def W(self, acap: float, bcap: float, ncap: int = 2):
        dij = np.eye(self.ndvr)
        nij = (1 - (1 - 1 / np.sqrt(2)) * dij)
        wij = self.wij(acap, bcap, ncap)
        wi = np.diag(wij)
        match self.spin:
            case "singlet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl *= nij; Cjl += Cjl.T
                    #Cik = nij * (np.matmul(wij, Cjl) + np.matmul(Cjl, wij.conj().T))
                    Cik = nij * (wi[:,None] * Cjl + Cjl * wi.conj()[None,:])
                    return Cik[self.map_ij, self.map_kl]
            case "triplet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl -= Cjl.T
                    Cik = (np.matmul(wij, Cjl) + np.matmul(Cjl, wij.conj().T))
                    return Cik[self.map_ij, self.map_kl]
        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, dtype=np.complex128)

    def Wold(self, acap: float, bcap: float, ncap: int = 2):
        dij = np.eye(self.ndvr)
        nij = (1 - (1 - 1 / np.sqrt(2)) * dij)
        wij = self.wij(acap, bcap, ncap)
        #wij = dvr_W(self.a, self.b, self.N, acap, bcap, ncap, self.bounds)
        match self.spin:
            case "singlet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl *= nij; Cjl += Cjl.T
                    Cik = nij * (np.matmul(wij, Cjl) + np.matmul(Cjl, wij.conj().T))
                    return Cik[self.map_ij, self.map_kl]
            case "triplet":
                def matvec(C):
                    Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl -= Cjl.T
                    Cik = (np.matmul(wij, Cjl) + np.matmul(Cjl, wij.conj().T))
                    return Cik[self.map_ij, self.map_kl]
        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, dtype=np.complex128)


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
        d2hij = self.d2hij(R)
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

#    def Hdvr_operator(self, R: float):
#        dij = np.eye(self.ndvr)
#        hij = self.hij(R)
#        Vik = self.Vik()
#        def matvec(CJ):
#            Cjl = np.zeros((self.ndvr, self.ndvr), dtype=np.complex128)
#            Cjl[self.map_ij, self.map_kl] = CJ
#            Cjl[self.map_kl, self.map_ij] = -CJ
#            #Cjl += -Cjl.swapaxes(0,1); Cjl *= 1 / np.sqrt(2)
#            #Cjl *= (1 - (1 - 1 / np.sqrt(2)) * dij)
#            #Cik = (1 - (1 - 1 / np.sqrt(2)) * dij) * (np.matmul(hij, Cjl) + np.matmul(Cjl, hij.conj().T) + Vik * Cjl)
#            Cik = (np.matmul(hij, Cjl) + np.matmul(Cjl, hij.conj().T) + Vik * Cjl)
#            #Cik = (hij @ Cjl + Cjl @ hij + Vik * Cjl)
#            CI = Cik[self.map_ij, self.map_kl]
#            return CI
#        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, dtype=np.complex128)

    def solve_wfn_dense(self, R: float, nbo: int = 0):
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
        d2Hnm = dvr_to_bo(Cijn, self.d2Hikjl(R))
        d2En = solve_d2En(En, d1Hnm, d2Hnm)
        np.savez('eigspec0', xi=self.xi(), En=En, Cijn=Cijn, d1En=d1En, d1Hnm=d1Hnm, nac1=nac1, d2En=d2En, d2Hnm=d2Hnm)
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

