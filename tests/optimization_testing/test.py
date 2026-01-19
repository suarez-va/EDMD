import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
import time

from grids.colbert_miller_dvr import dvr_x, dvr_T, dvr_W

def linearoperator_to_dense(H_op):
    n, m = H_op.shape
    assert n == m

    H = np.zeros((n, n), dtype=H_op.dtype)

    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        H[:, j] = H_op @ e

    return H

def HIJ(n):
    N = int(n*(n-1)/2)
    H = np.zeros((N, N), dtype=np.complex128)
    map_ij, map_kl = np.triu_indices(n, k=1)
    Hikjl = np.zeros((n, n, n, n), dtype=np.complex128)
    xi = np.arange(n)
    dij = np.eye(n)
    #hij = (1.+ 0j) * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    hij = dvr_T(1.0, -25.0, 25.0, n-1, "(-inf,inf)")
    print(np.max(np.absolute(hij - hij.conj().T)))
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
    #hij = (1.+ 0j) * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    hij = dvr_T(1.0, -25.0, 25.0, n-1, "(-inf,inf)")
    Vik = np.exp(-(xi[:,None] - xi[None,:])**2)

    def HC(c):
        Ckl = np.zeros((n, n), dtype=np.complex128)
        Ckl[map_ij, map_kl] = c
        Ckl[map_kl, map_ij] = -c
        Cij = (hij @ Ckl + Ckl @ hij + Vik * Ckl)
        #Cij = (Cij - Cij.conj().T) / np.sqrt(2)
        return Cij[map_ij, map_kl]

    return LinearOperator(shape=(N, N), matvec=HC, dtype=np.complex128)

#nsub=17
#
#H = HIJ(nsub)
##print(np.max(np.absolute(H-H.T)))
#time1 = time.time()
#vals1, vecs1 = eigsh(H, k=125, which='SA')
#idx = np.argsort(vals1)
#vals1 = vals1[idx]; vecs1 = vecs1[:,idx]
#time2 = time.time()
##print(time2-time1)
#
#Hop = make_H_operator(nsub)
#time3 = time.time()
#vals2, vecs2 = eigsh(Hop, k=125, which='SA')
#idx = np.argsort(vals2)
#vals2 = vals2[idx]; vecs2 = vecs2[:,idx]
#time4 = time.time()
#print(time4-time3)

#print(vals1 - vals2)

#print(np.max(np.absolute(linearoperator_to_dense(Hop) - H)))
#exit()


#from models.model_utils import dvr_to_bo
from models.two_electron_diatomic import TEGD

R_sub = 8

params = {
    "aR": 0.0,
    "bR": 0.0001,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.0,
    "bee": 0.0001,
}

keig = 125

#model = TEGD(a=-12.5, b=12.5, N=20, bounds="(-inf,inf)", spin="triplet", model_params=params)
model = TEGD(a=-25.0, b=25.0, N=50, bounds="(-inf,inf)", spin="singlet", model_params=params)

model.solve_wfn_dense(R = R_sub, nbo = keig)
model.solve_wfn(R = R_sub, nbo = keig)

eigspec0 = np.load("eigspec0.npz")
eigspec = np.load("eigspec.npz")

print(np.max(np.absolute(eigspec0["En"] - eigspec["En"])))
print(np.max(np.absolute(eigspec0["d1En"] - eigspec["d1En"])))
print(np.max(np.absolute(eigspec0["d2En"] - eigspec["d2En"])))
Cijn0 = eigspec0["Cijn"]
Cijn = eigspec["Cijn"]

C0 = Cijn0.reshape(model.ndvr**2, keig)
C = Cijn.reshape(model.ndvr**2, keig)

ovlp = np.absolute(np.matmul(C.conj().T, C))
print(np.round(ovlp,10))
#print(np.matmul(C0.conj().T, C0))
print()


exit()



ep, cip = model.solve_mos(R = R_sub)
vals0 = ep[model.map_ij] + ep[model.map_kl]
idx = np.argsort(vals0)
vals0 = vals0[idx]
vals0 = vals0[:keig]

Hdvr = model.Hdvr(R = R_sub)
time5 = time.time()
#vals3, vecs3 = eigsh(Hdvr, k=125, which='SA', ncv=8*125, tol=1e-12)
vals3, vecs3 = eigsh(Hdvr, k=keig, which='SA')
idx = np.argsort(vals3)
vals3 = vals3[idx]; vecs3 = vecs3[:,idx]
time6 = time.time()
print(time6-time5)

Hdvrop = model.Hdvr_operator(R = R_sub)
time7 = time.time()
#vals4, vecs4 = eigsh(Hdvrop, k=125, which='SA', ncv=8*125, tol=1e-12)
vals4, vecs4 = eigsh(Hdvrop, k=keig, which='SA')
idx = np.argsort(vals4)
vals4 = vals4[idx]; vecs4 = vecs4[:,idx]
time8 = time.time()
print(time8-time7)

print(f"mos - wfn1: {vals0 - vals3}")
print(f"mos - wfn2: {vals0 - vals4}")
print(f"wfn1 - wfn2: {vals3 - vals4}")

#print(np.absolute(vals4-vals3)/np.absolute(vals3))
#imax = np.argmax(np.absolute(vals4-vals3)/np.absolute(vals3))
#print(imax)
#print(vals3[imax])
#print(vals4[imax])

exit()

Hdense = linearoperator_to_dense(Hdvrop)
time9 = time.time()
vals5, vecs5 = eigsh(Hdense, k=125, which='SA', ncv=8*125, tol=1e-12)
idx = np.argsort(vals5)
vals5 = vals5[idx]; vecs5 = vecs5[:,idx]
time10 = time.time()
print(time10-time9)
print("3 vs 4:")
print((vals4-vals3)/np.absolute(vals3))
print("3 vs 5:")
print((vals5-vals3)/np.absolute(vals3))
print("4 vs 5:")
print((vals5-vals4)/np.absolute(vals4))

vals6, vecs6 = eigsh(Hdvr, k=125, which='SA', ncv=8*125, tol=1e-12)
idx = np.argsort(vals6)
vals6 = vals6[idx]; vecs6 = vecs6[:,idx]
print("3 vs 6:")
print((vals6-vals3)/np.absolute(vals6))



#print(vals4-vals3)

#Wab20n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-20.0, bcap=20.0, ncap=2))
#Wab40n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-40.0, bcap=40.0, ncap=2))
#Wab50n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-50.0, bcap=50.0, ncap=2))
#Wab60n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-60.0, bcap=60.0, ncap=2))
#Wab80n2 = dvr_to_bo(Cijn, model.Wikjl(acap=-80.0, bcap=80.0, ncap=2))
#np.savez("boops", Wab20n2=Wab20n2, Wab40n2=Wab40n2, Wab50n2=Wab50n2, Wab60n2=Wab60n2, Wab80n2=Wab80n2)

exit()
print("problems here:")
Hdense = linearoperator_to_dense(Hdvrop)
print(Hdense[0,13])
print(Hdvr[0,13])

Herror = Hdense - Hdvr

#print(Herror.shape)
print(np.min(np.absolute(Herror)))
print(np.max(np.absolute(Herror)))
#print(np.argwhere(np.absolute(Herror) >= 0.001))

#print(model.map_ij[13], model.map_kl[13])
#for I in range(model.nfci):
#    print(model.map_ij[I], model.map_kl[I])

