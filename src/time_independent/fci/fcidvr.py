from typing import Optional
import itertools
import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigsh, LinearOperator
from grids.colbert_miller_dvr import dvr_T
from model_systems.models import Model
from time_independent.utils import solve_nac1, solve_d2En, solve_nac2

class FCIDVR:
    def __init__(self, model: Model, xa: float, xb: float, xN: int, xbounds: str):
        self.model = model
        self.xa = xa
        self.xb = xb
        self.xN = xN
        self.xbounds = xbounds
        assert self.xbounds in ('(a,b)', '(0,inf)', '(-inf,inf)'), f"Colber Miller Electronic DVR Bounds: {self.xbounds}, must be '(a,b)', '(0,inf)', or '(-inf,inf)'."

        self.nxdvr = int(self.xN - 1) if self.xbounds=='(a,b)' else int(self.xN) if self.xbounds=='(0,inf)' else int(self.xN + 1) if self.xbounds=='(-inf,inf)' else 0

    def xi(self) -> np.ndarray:
        return np.linspace(self.xa, self.xb, self.nxdvr)

    # generate hcore using Colbert-Miller syle DVR
    def hij(self, R: float) -> np.ndarray:
        return dvr_T(1, self.xa, self.xb, self.xN, self.xbounds) + np.diag(self.model.VeR(self.xi(), R))

    # generate derivative of hcore with respect to R using Colbert-Miller syle DVR for kinetic energy
    def d1hij(self, R: float) -> np.ndarray:
        return np.diag(self.model.d1VeR(self.xi(), R))

    def d2hij(self, R: float) -> np.ndarray:
        return np.diag(self.model.d2VeR(self.xi(), R))

    # generate cap using Colbert-Miller syle DVR
    def wi(self, acap: float, bcap: float, ncap: int = 2) -> np.ndarray:
        xi=self.xi()
        return (xi - bcap)**ncap * np.heaviside(xi - bcap, 0.5) + np.abs(xi - acap)**ncap * np.heaviside(-(xi - acap), 0.5)

    def Vik(self) -> np.ndarray:
        xi=self.xi()
        return self.model.Vee(xi[:,None],xi[None,:])

    def validate_nele_spin(self, nele: int, spin: str):
        allowed_cases = {
            1: ['doublet'],
            2: ['singlet', 'triplet'],
            3: ['doublet']
        }
        if nele not in allowed_cases:
            raise ValueError(f"FCIDVR not implemented for nele = {nele}")
        if spin not in allowed_cases[nele]:
            raise ValueError(f"Invalid spin '{spin}' for nele = {nele}.\n Allowed spins: {allowed_cases[nele]}")
        return None

    def CImapping(self, nele: int, spin: str):
        self.validate_nele_spin(nele, spin)
        if nele == 1:
            match spin:
                case 'doublet':
                    nfci = self.nxdvr
                    nmap = np.arange(self.nxdvr, dtype = np.float64)#; nmap = np.ones(self.nxdvr, dtype = np.float64)
                    CImap = np.arange(self.nxdvr, dtype = int)#; CImap = np.ones(self.nxdvr, dtype = int)
                    return nfci, nmap, CImap
        if nele == 2:
            match spin:
                case 'singlet':
                    nfci = int((self.nxdvr + 1) * self.nxdvr / 2)
                    nmap = np.triu((1 - (1 - 1 / np.sqrt(2)) * np.eye(self.nxdvr)), k = 0)
                    CImap = np.triu_indices(self.nxdvr, k = 0)
                    return nfci, nmap, CImap
                case 'triplet':
                    nfci = int(self.nxdvr * (self.nxdvr - 1) / 2)
                    nmap = np.triu(np.zeros((self.nxdvr, self.nxdvr)), k = 1)
                    CImap = np.triu_indices(self.nxdvr, k = 1)
                    return nfci, nmap, CImap
        if nele == 3:
            match spin:
                case 'doublet':
                    nfci = int((self.nxdvr + 1) * self.nxdvr * (self.nxdvr - 1) / 3)
                    i, j, k = np.indices((self.nxdvr, self.nxdvr, self.nxdvr))
                    nmap1 = np.zeros((self.nxdvr, self.nxdvr, self.nxdvr)); nmap1[(i < j) & (j < k)] = 1.0; nmap1[(i < j) & (j == k)] = np.sqrt(0.5)
                    nmap2 = np.zeros((self.nxdvr, self.nxdvr, self.nxdvr)); nmap2[(i < j) & (j < k)] = 1.0; nmap2[(i == j) & (j < k)] = np.sqrt(2.0/3.0)
                    nmap = (nmap1, nmap2)
                    CImap1 = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(self.nxdvr), 3) if (i < j < k) or (i < j and j == k)]))
                    CImap2 = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(self.nxdvr), 3) if (i < j < k) or (i == j and j < k)]))
                    CImap = (CImap1, CImap2)
                    return nfci, nmap, CImap
        assert False

    def CIoperator(self, nele: int, spin: str, hij: np.ndarray, gik: Optional[np.ndarray] = None):
        self.validate_nele_spin(nele, spin)
        nfci, nmap, CImap = self.CImapping(nele, spin)
        gik = np.zeros((self.nxdvr, self.nxdvr)) if gik is None else gik
        matvec = lambda C: C
        if nele == 1:
            match spin:
                case 'doublet':
                    def matvec(C):
                        return hij @ C
        if nele == 2:
            match spin:
                case 'singlet':
                    def matvec(C):
                        Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype = np.complex128)
                        Cjl[CImap] = C; Cjl *= nmap; Cjl += Cjl.T
                        hC = hij @ Cjl
                        Cik = nmap * (hC + hC.T + gik * Cjl)
                        return Cik[CImap]
                case 'triplet':
                    def matvec(C):
                        Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype = np.complex128)
                        Cjl[CImap] = C; Cjl -= Cjl.T
                        hC = hij @ Cjl
                        Cik = (hC + hC.T + gik * Cjl)
                        return Cik[CImap]
        if nele == 3:
            match spin:
                case 'doublet':
                    def matvec(C):
                        Cjln1 = np.zeros((self.nxdvr, self.nxdvr, self.nxdvr), dtype = np.complex128); Cjln1[CImap[0]] = C[:int(nfci/2)].copy(); nC1 = nmap[0] * Cjln1
                        Cjln2 = np.zeros((self.nxdvr, self.nxdvr, self.nxdvr), dtype = np.complex128); Cjln2[CImap[1]] = C[int(nfci/2):].copy(); nC2 = nmap[1] * Cjln2
                        gikm = gik[:,:,None] + gik[:,None,:] + gik[None,:,:]
                        gC1 = gikm*Cjln1; gC2 = gikm*Cjln2
                        Cjln1 = nC1 + nC1.swapaxes(1,2) - 0.5*(nC1.swapaxes(0,1) + nC1.transpose(2,0,1) + nC1.transpose(1,2,0) + nC1.swapaxes(0,2)) + 0.5*np.sqrt(3)*(nC2.swapaxes(0,1) - nC2.transpose(2,0,1) + nC2.transpose(1,2,0) - nC2.swapaxes(0,2))
                        Cjln2 = nC2 - nC2.swapaxes(1,2) + 0.5*(nC2.swapaxes(0,1) - nC2.transpose(2,0,1) - nC2.transpose(1,2,0) + nC2.swapaxes(0,2)) + 0.5*np.sqrt(3)*(nC1.swapaxes(0,1) + nC1.transpose(2,0,1) - nC1.transpose(1,2,0) - nC1.swapaxes(0,2))
                        hC1 = (hij @ Cjln1.reshape(self.nxdvr, self.nxdvr**2)).reshape(self.nxdvr, self.nxdvr, self.nxdvr); Ch1 = (Cjln1.reshape(self.nxdvr**2, self.nxdvr) @ hij.conj()).reshape(self.nxdvr, self.nxdvr, self.nxdvr)
                        hC2 = (hij @ Cjln2.reshape(self.nxdvr, self.nxdvr**2)).reshape(self.nxdvr, self.nxdvr, self.nxdvr); Ch2 = (Cjln2.reshape(self.nxdvr**2, self.nxdvr) @ hij.conj()).reshape(self.nxdvr, self.nxdvr, self.nxdvr)
                        Cikm1 = nmap[0]*(hC1 + Ch1.swapaxes(1,2) + Ch1) + gC1
                        Cikm2 = nmap[1]*(hC2 - Ch2.swapaxes(1,2) + Ch2) + gC2
                        return np.concatenate((Cikm1[CImap[0]], Cikm2[CImap[1]]))
        def matmat(Cm):
            m = Cm.shape[1]
            Cn = np.zeros_like(Cm, dtype=np.complex128)
            for j in range(m):
                Cn[:, j] = matvec(Cm[:, j])
            return Cn
        return LinearOperator(shape=(nfci, nfci), matvec=matvec, matmat=matmat, dtype=np.complex128)

    def kernel(self, R: float, nele: int = 1, spin: str = 'doublet', nbo: int = 100):
        self.validate_nele_spin(nele, spin)
        xi = self.xi()
        hij = self.hij(R)
        Vik = self.Vik()
        H = self.CIoperator(nele, spin, hij, Vik)

        En, Cn = eigsh(H, k = nbo, which = 'SA')
        idx = np.argsort(En)
        En = En[idx]; Cn = Cn[:,idx]
        Cn *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,:])

        X = self.CIoperator(nele, spin, np.diag(xi))
        d1H = self.CIoperator(nele, spin, self.d1hij(R))
        d2H = self.CIoperator(nele, spin, self.d2hij(R))

        xnm = Cn.conj().T @ X.matmat(Cn)
        En += self.model.VR(R)
        d1Hnm = Cn.conj().T @ d1H.matmat(Cn) + self.model.d1VR(R) * np.eye(nbo)
        d1En = np.diag(d1Hnm.real)
        nac1 = solve_nac1(En, d1Hnm)
        d2Hnm = Cn.conj().T @ d2H.matmat(Cn) + self.model.d2VR(R) * np.eye(nbo)
        d2En = solve_d2En(En, d1Hnm, d2Hnm)
        nac2 = solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)

        np.savez(f'fcidvr_{nele}ele_{spin}', xi=xi, En=En, Cn=Cn, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
        return self

####################################################################################################################################################################

#    def HamiltonianOperator(self, R: float, nele, spin: str):
#        self.validate_nele_spin(nele, spin)
#        nfci, nmap, CImap = self.CImapping(nele, spin)
#        hij = self.hij(R)
#        Vik = self.Vik()
#        if nele == 1:
#            match spin:
#                case 'doublet':
#                    def matvec(C):
#                        return hij @ C
#        if nele == 2:
#            match spin:
#                case 'singlet':
#                    def matvec(C):
#                        Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
#                        Cjl[CImap] = C; Cjl *= nmap; Cjl += Cjl.T
#                        hC = hij @ Cjl
#                        Cik = nmap * (hC + hC.T + Vik * Cjl)
#                        return Cik[CImap]
#                case 'triplet':
#                    def matvec(C):
#                        Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
#                        Cjl[CImap] = C; Cjl -= Cjl.T
#                        hC = hij @ Cjl
#                        Cik = (hC + hC.T + Vik * Cjl)
#                        return Cik[CImap]
#        def matmat(Cn):
#            n = Cn.shape[1]
#            res = np.zeros_like(Cn, dtype=np.complex128)
#            for j in range(n):
#                res[:, j] = matvec(Cn[:, j])
#            return res
#        return LinearOperator(shape=(nfci, nfci), matvec=matvec, matmat=matmat, dtype=np.complex128)

#    def kernel(self, R: float, nele: int = 1, spin: str = 'singlet', nbo: int = 100):
#        self.validate_nele_spin(nele, spin)
#
#        xi = self.xi()
#        En, Cn = eigsh(self.HamiltonianOperator(R), k=nbo, which='SA')
#        idx = np.argsort(En)
#        En = En[idx]; Cn = Cn[:,idx]; Cijn = self.vec_to_wfn(Cn)
#        Cijn *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,None,:])
#
#        xnm = Cn.conj().T @ self.CustomOperator(np.diag(xi)).matmat(Cn)
#        En += self.model.VR(R)
#        d1Hnm = Cn.conj().T @ self.CustomOperator(self.d1hij(R)).matmat(Cn) + self.model.d1VR(R) * np.eye(nbo)
#        d1En = np.diag(d1Hnm.real)
#        nac1 = self.solve_nac1(En, d1Hnm)
#        d2Hnm = Cn.conj().T @ self.CustomOperator(self.d2hij(R)).matmat(Cn) + self.model.d2VR(R) * np.eye(nbo)
#        d2En = self.solve_d2En(En, d1Hnm, d2Hnm)
#        nac2 = self.solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)
#
#        np.savez('2efci', xi=xi, En=En, Cijn=Cijn, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
#        return self

#    def solve_fci(self, R: float, nbo: int = 100):
#        xi = self.xi()
#        En, Cin = eigh(self.hij(R), subset_by_index = (0, nbo - 1))
#        Cin *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,:])
#
#        xnm = Cin.conj().T @ (xi[:,None] * Cin)
#        En += self.model.VR(R)
#        d1Hnm = Cin.conj().T @ self.d1hij(R) @ Cin + self.model.d1VR(R) * np.eye(nbo)
#        d1En = np.diag(d1Hnm.real)
#        nac1 = self.solve_nac1(En, d1Hnm)
#        d2Hnm = Cin.conj().T @ self.d2hij(R) @ Cin + self.model.d2VR(R) * np.eye(nbo)
#        d2En = self.solve_d2En(En, d1Hnm, d2Hnm)
#        nac2 = self.solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)
#
#        np.savez('1efci', xi=xi, En=En, Cin=Cin, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
#        return self

