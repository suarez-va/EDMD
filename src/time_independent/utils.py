import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigsh, LinearOperator
from grids.colbert_miller_dvr import dvr_x, dvr_T, dvr_W
from model_systems.models import Model

def solve_nac1(En, d1Hnm):
    nadi = En.shape[0]
    nac1 = np.zeros((nadi, nadi), dtype=np.complex128)
    mask = np.ones((nadi, nadi), dtype=bool)
    np.fill_diagonal(mask, False)
    DE = En[:, None] - En[None, :]
    nac1[mask] = -d1Hnm[mask] / DE[mask]
    return nac1

def solve_d2En(En, d1Hnm, d2Hnm):
    nadi = En.shape[0]
    d2En = np.zeros((nadi), dtype=np.float64)
    M = np.zeros((nadi, nadi), dtype=np.float64)
    mask = np.ones((nadi, nadi), dtype=bool)
    np.fill_diagonal(mask, False)
    DE = En[:, None] - En[None, :]
    M[mask] = np.absolute(d1Hnm[mask])**2 / DE[mask]
    d2En = np.diagonal(d2Hnm.real) + 2 * np.sum(M, axis=1)
    return d2En

def solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm):
    nadi = En.shape[0]
    nac2 = np.zeros((nadi, nadi), dtype=np.complex128)
    mask = np.ones((nadi, nadi), dtype=bool)
    np.fill_diagonal(mask, False)
    DE = En[:, None] - En[None, :]
    Dd1E = d1En[:, None] - d1En[None, :]
    nac2 += nac1 @ nac1
    nac2[mask] += -((d1Hnm @ nac1 - nac1 @ d1Hnm) + Dd1E * nac1 + d2Hnm)[mask] / DE[mask]
    return nac2

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

def matmat(operator, vectors):
    k = vectors.shape[1]
    out = np.zeros_like(vectors, dtype=np.complex128)
    for j in range(k):
        out[:, j] = operator.matvec(vectors[:, j])
    return out



def dvr_to_bo(Cijn, Oikjl):
    ndvr = Cijn.shape[0]
    nbo = Cijn.shape[-1]
    CNn = Cijn.reshape(ndvr**2, nbo)
    ONM = Oikjl.reshape(ndvr**2, ndvr**2)
    Onm = np.linalg.multi_dot([CNn.conj().T, ONM, CNn])
    return Onm
