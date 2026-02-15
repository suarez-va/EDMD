import numpy as np
from scipy.linalg import eigh
from grids.colbert_miller_dvr import dvr_T
from model_systems.models import Model


class OneElectronFixedNuclei:
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

    def solve_nac1(self, En, d1Hnm):
        nadi = En.shape[0]
        nac1 = np.zeros((nadi, nadi), dtype=np.complex128)
        mask = np.ones((nadi, nadi), dtype=bool)
        np.fill_diagonal(mask, False)
        DE = En[:, None] - En[None, :]
        nac1[mask] = -d1Hnm[mask] / DE[mask]
        return nac1

    def solve_d2En(self, En, d1Hnm, d2Hnm):
        nadi = En.shape[0]
        d2En = np.zeros((nadi), dtype=np.float64)
        M = np.zeros((nadi, nadi), dtype=np.float64)
        mask = np.ones((nadi, nadi), dtype=bool)
        np.fill_diagonal(mask, False)
        DE = En[:, None] - En[None, :]
        M[mask] = np.absolute(d1Hnm[mask])**2 / DE[mask]
        d2En = np.diagonal(d2Hnm.real) + 2 * np.sum(M, axis=1)
        return d2En

    def solve_nac2(self, En, d1Hnm, d1En, nac1, d2Hnm):
        nadi = En.shape[0]
        nac2 = np.zeros((nadi, nadi), dtype=np.complex128)
        mask = np.ones((nadi, nadi), dtype=bool)
        np.fill_diagonal(mask, False)
        DE = En[:, None] - En[None, :]
        Dd1E = d1En[:, None] - d1En[None, :]
        nac2 += nac1 @ nac1
        nac2[mask] += -((d1Hnm @ nac1 - nac1 @ d1Hnm) + Dd1E * nac1 + d2Hnm)[mask] / DE[mask]
        return nac2

    def solve_fci(self, R: float, nbo: int = 100):
        xi = self.xi()
        En, Cin = eigh(self.hij(R), subset_by_index = (0, nbo - 1))
        Cin *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,:])

        xnm = Cin.conj().T @ (xi[:,None] * Cin)
        En += self.model.VR(R)
        d1Hnm = Cin.conj().T @ self.d1hij(R) @ Cin + self.model.d1VR(R) * np.eye(nbo)
        d1En = np.diag(d1Hnm.real)
        nac1 = self.solve_nac1(En, d1Hnm)
        d2Hnm = Cin.conj().T @ self.d2hij(R) @ Cin + self.model.d2VR(R) * np.eye(nbo)
        d2En = self.solve_d2En(En, d1Hnm, d2Hnm)
        nac2 = self.solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)

        np.savez('1efci', xi=xi, En=En, Cin=Cin, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
        return self

#    def solve_nac1(self, ep, d1hpq):
#        nadi = ep.shape[0]
#        nac1 = np.zeros((nadi, nadi), dtype=np.complex128)
#        mask = np.ones((nadi, nadi), dtype=bool)
#        np.fill_diagonal(mask, False)
#        De = ep[:, None] - ep[None, :]
#        nac1[mask] = -d1hpq[mask] / De[mask]
#        return nac1
#
#    def solve_d2ep(self, ep, d1hpq, d2hpq):
#        nadi = ep.shape[0]
#        d2ep = np.zeros((nadi), dtype=np.float64)
#        M = np.zeros((nadi, nadi), dtype=np.float64)
#        mask = np.ones((nadi, nadi), dtype=bool)
#        np.fill_diagonal(mask, False)
#        De = ep[:, None] - ep[None, :]
#        M[mask] = np.absolute(d1hpq[mask])**2 / De[mask]
#        d2ep = np.diagonal(d2hpq.real) + 2 * np.sum(M, axis=1)
#        return d2ep
#
#    def solve_nac2(self, ep, d1hpq, d1ep, nac1, d2hpq):
#        nadi = ep.shape[0]
#        nac2 = np.zeros((nadi, nadi), dtype=np.complex128)
#        mask = np.ones((nadi, nadi), dtype=bool)
#        np.fill_diagonal(mask, False)
#        De = ep[:, None] - ep[None, :]
#        Dd1e = d1ep[:, None] - d1ep[None, :]
#        nac2 += nac1 @ nac1
#        nac2[mask] += -((d1hpq @ nac1 - nac1 @ d1hpq) + Dd1e * nac1 + d2hpq)[mask] / De[mask]
#        return nac2
#
#
#
#    def solve_mos(self, R: float, nmo: int = 100):
#        xi = self.xi()
#        ep, cip = eigh(self.hij(R), subset_by_index = (0, nmo - 1))
#        cip *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nmo)[None,:])
#
#        xpq = cip.conj().T @ (xi[:,None] * cip)
#        ep += self.model.VR(R)
#        d1hpq = cip.conj().T @ self.d1hij(R) @ cip + self.model.d1VR(R) * np.eye(nmo)
#        d1ep = np.diag(d1hpq.real)
#        nac1 = self.solve_nac1(ep, d1hpq)
#        d2hpq = cip.conj().T @ self.d2hij(R) @ cip + self.model.d2VR(R) * np.eye(nmo)
#        d2ep = self.solve_d2ep(ep, d1hpq, d2hpq)
#        nac2 = self.solve_nac2(ep, d1hpq, d1ep, nac1, d2hpq)
#
#        np.savez('mospec', xi=xi, ep=ep, cip=cip, xpq=xpq, d1hpq=d1hpq, d1ep=d1ep, nac1=nac1, d2hpq=d2hpq, d2ep=d2ep, nac2=nac2)
#        return self

