import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
import time

def HIJ(n):
    N = int(n*(n-1)/2)
    H = np.zeros((N, N), dtype=np.float64)
    map_ij, map_kl = np.triu_indices(n, k=1)
    Hikjl = np.zeros((n, n, n, n), dtype=np.float64)
    xi = np.arange(n)
    dij = np.eye(n)
    hij = (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    Vik = np.exp(-(xi[:,None] - xi[None,:])**2)
    Hikjl += hij[:,None,:,None] * dij[None,:,None,:]
    Hikjl += dij[:,None,:,None] * hij[None,:,None,:]
    Hikjl += Vik[:,:,None,None] * dij[:,None,:,None] * dij[None,:,None,:]
    H += Hikjl[map_ij[:,None], map_kl[:,None], map_ij[None,:], map_kl[None,:]]
    H -= Hikjl[map_ij[:,None], map_kl[:,None], map_kl[None,:], map_ij[None,:]]
    return H

def make_H_operator(n):
    N = int(n*(n-1)/2)
    map_ij, map_kl = np.triu_indices(n, k=1)
    xi = np.arange(n)
    hij = 2*np.eye(n) - np.eye(n,k=1) - np.eye(n,k=-1)
    Vik = np.exp(-(xi[:,None] - xi[None,:])**2)

    def HC(c):
        Ckl = np.zeros((n, n))
        Ckl[map_ij, map_kl] = c
        Ckl[map_kl, map_ij] = -c
        Cij = (hij @ Ckl + Ckl @ hij + Vik * Ckl)
        #Cij = Cij - Cij.T
        return Cij[map_ij, map_kl]

    return LinearOperator(shape=(N, N), matvec=HC, dtype=np.float64)

nsub=100

H = HIJ(nsub)
#print(np.max(np.absolute(H-H.T)))
time1 = time.time()
vals1, vecs1 = eigsh(H, k=125, which='SA')
time2 = time.time()
print(time2-time1)

Hop = make_H_operator(nsub)
time3 = time.time()
vals2, vecs2 = eigsh(Hop, k=125, which='SA')
time4 = time.time()
print(time4-time3)

print(vals1 - vals2)
