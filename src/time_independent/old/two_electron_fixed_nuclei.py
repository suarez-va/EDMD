import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigsh, LinearOperator
from grids.colbert_miller_dvr import dvr_x, dvr_T, dvr_W
from model_systems.models import Model
from time_independent.fci.one_electron_fixed_nuclei import OneElectronFixedNuclei


class TwoElectronFixedNuclei(OneElectronFixedNuclei):
    def __init__(self, model: Model, xa: float, xb: float, xN: int, xbounds: str, spin: str):

        super().__init__(model, xa, xb, xN, xbounds)

        self.spin = spin
        assert self.spin in ("singlet", "triplet"), f"Spin Multiplicity: {self.spin}, must be 'singlet' or 'triplet'."

        self.nfci = int((self.nxdvr+1)*self.nxdvr/2) if self.spin=="singlet" else int(self.nxdvr*(self.nxdvr-1)/2) if self.spin=="triplet" else 0
        self.map_ij, self.map_kl = np.triu_indices(self.nxdvr, k=0) if self.spin=="singlet" else np.triu_indices(self.nxdvr, k=1) if self.spin=="triplet" else np.zeros((self.nfci), dtype=int)

    def Vik(self) -> np.ndarray:
        xi=self.xi()
        return self.model.Vee(xi[:,None],xi[None,:])

    def wfn_to_vec(self, Cijn):
        assert (Cijn.shape[0] == self.nxdvr) and (Cijn.shape[1] == self.nxdvr), f"First two dimensions of wavefunction Cijn, must match electronic DVR dimensionality: ({self.nxdvr}, {self.nxdvr})"
        assert (Cijn.ndim == 2) or (Cijn.ndim == 3), f"Wavefunction Cijn should have either 2 or 3 dimensions"
        if Cijn.ndim==2:
            nbo = 0
            Cijn = Cijn[:,:,None]
            Cn = np.zeros((self.nfci, 1), dtype=np.complex128)
        else:
            nbo = Cijn.shape[2]
            Cn = np.zeros((self.nfci, nbo), dtype=np.complex128)
        match self.spin:
            case "singlet":
                nij = (1 - (1 - 1 / np.sqrt(2)) * np.eye(self.nxdvr))
                Cn = nij[self.map_ij, self.map_kl, None] / np.sqrt(2) * (Cijn[self.map_ij, self.map_kl, :] + Cijn[self.map_kl, self.map_ij, :])
            case "triplet":
                Cn = 1 / np.sqrt(2) * (Cijn[self.map_ij, self.map_kl, :] - Cijn[self.map_kl, self.map_ij, :])
        return Cn[:, 0] if (nbo==0) else Cn

    def vec_to_wfn(self, Cn):
        assert (Cn.shape[0] == self.nfci), f"First dimension of CI vector Cn, must match FCI dimensionality: ({self.nfci})"
        assert (Cn.ndim == 1) or (Cn.ndim == 2), f"CI vector Cn should have either 1 or 2 dimensions"
        if Cn.ndim==1:
            nbo = 0
            Cn = Cn[:,None]
            Cijn = np.zeros((self.nxdvr, self.nxdvr, 1), dtype=np.complex128)
        else:
            nbo = Cn.shape[1]
            Cijn = np.zeros((self.nxdvr, self.nxdvr, nbo), dtype=np.complex128)
        Cijn[self.map_ij, self.map_kl, :] = Cn
        match self.spin:
            case "singlet":
                nij = (1 - (1 - 1 / np.sqrt(2)) * np.eye(self.nxdvr))
                Cijn *= nij[:,:,None]
                Cijn += Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
            case "triplet":
                Cijn += -Cijn.swapaxes(0,1)
                Cijn *= 1 / np.sqrt(2)
        return Cijn[:, :, 0] if (nbo==0) else Cijn

    def HamiltonianOperator(self, R: float):
        nij = (1 - (1 - 1 / np.sqrt(2)) * np.eye(self.nxdvr))
        hij = self.hij(R)
        Vik = self.Vik()
        match self.spin:
            case "singlet":
                def matvec(C):
                    Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl *= nij; Cjl += Cjl.T
                    hC = hij @ Cjl
                    Cik = nij * (hC + hC.T + Vik * Cjl)
                    #Cik = nij * (hij @ Cjl + Cjl @ hij.conj().T + Vik * Cjl)
                    return Cik[self.map_ij, self.map_kl]
            case "triplet":
                def matvec(C):
                    Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl -= Cjl.T
                    hC = hij @ Cjl
                    Cik = (hC + hC.T + Vik * Cjl)
                    #Cik = (hij @ Cjl + Cjl @ hij.conj().T + Vik * Cjl)
                    return Cik[self.map_ij, self.map_kl]
        def matmat(Cn):
            n = Cn.shape[1]
            res = np.zeros_like(Cn, dtype=np.complex128)
            for j in range(n):
                res[:, j] = matvec(Cn[:, j])
            return res
        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, matmat=matmat, dtype=np.complex128)

    def CustomOperator(self, oij):
        nij = (1 - (1 - 1 / np.sqrt(2)) * np.eye(self.nxdvr))
        match self.spin:
            case "singlet":
                def matvec(C):
                    Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl *= nij; Cjl += Cjl.T
                    Cik = nij * (oij @ Cjl + Cjl @ oij.conj().T)
                    return Cik[self.map_ij, self.map_kl]
            case "triplet":
                def matvec(C):
                    Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
                    Cjl[self.map_ij, self.map_kl] = C; Cjl -= Cjl.T
                    Cik = (oij @ Cjl + Cjl @ oij.conj().T)
                    return Cik[self.map_ij, self.map_kl]
        def matmat(Cn):
            n = Cn.shape[1]
            res = np.zeros_like(Cn, dtype=np.complex128)
            for j in range(n):
                res[:, j] = matvec(Cn[:, j])
            return res
        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, matmat=matmat, dtype=np.complex128)

    def solve_fci(self, R: float, nbo: int = 100):
        xi = self.xi()
        En, Cn = eigsh(self.HamiltonianOperator(R), k=nbo, which='SA')
        idx = np.argsort(En)
        En = En[idx]; Cn = Cn[:,idx]; Cijn = self.vec_to_wfn(Cn)
        Cijn *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,None,:])

        xnm = Cn.conj().T @ self.CustomOperator(np.diag(xi)).matmat(Cn)
        En += self.model.VR(R)
        d1Hnm = Cn.conj().T @ self.CustomOperator(self.d1hij(R)).matmat(Cn) + self.model.d1VR(R) * np.eye(nbo)
        d1En = np.diag(d1Hnm.real)
        nac1 = self.solve_nac1(En, d1Hnm)
        d2Hnm = Cn.conj().T @ self.CustomOperator(self.d2hij(R)).matmat(Cn) + self.model.d2VR(R) * np.eye(nbo)
        d2En = self.solve_d2En(En, d1Hnm, d2Hnm)
        nac2 = self.solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)

        np.savez('2efci', xi=xi, En=En, Cijn=Cijn, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
        return self

