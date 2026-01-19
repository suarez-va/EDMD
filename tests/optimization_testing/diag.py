import numpy as np
from scipy.sparse.linalg import eigsh
from models.two_electron_diatomic import TEGD
import time

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

#model = TEGD(a=-192.5, b=192.5, N=500, bounds="(-inf,inf)", spin="triplet", model_params=params)
model = TEGD(a=-50.0, b=50.0, N=125, bounds="(-inf,inf)", spin="triplet", model_params=params)

time1 = time.time()
CIn = model.solve_wfn(R = R_sub, nbo = 250)
time2 = time.time()
print(time2-time1)

exit()

Hdvrop = model.H(R = R_sub)
#vals4, vecs4 = eigsh(Hdvrop, k=125, which='SA', ncv=8*125, tol=1e-12)
vals, vecs = eigsh(Hdvrop, k=125, which='SA')
idx = np.argsort(vals)
vals = vals[idx]; vecs = vecs[:,idx]
time2 = time.time()
print(time2-time1)

def matmat(operator, vectors):
    k = vectors.shape[1]
    out = np.zeros_like(vectors, dtype=np.complex128)
    for j in range(k):
        out[:, j] = operator.matvec(vectors[:, j])
    return out

print(vecs.shape)
HIm = matmat(Hdvrop, vecs)
print(HIm.shape)
Hnm = np.matmul(vecs.conj().T, HIm)
print(Hnm.shape)
print(np.diag(Hnm.real) - vals)
print(np.diag(Hnm.real))
print(np.max(np.absolute(Hnm.real - np.diag(vals))))


#V = np.column_stack(vecs)
#print(V.shape)
#Hnm = V.conj().T @ (Hdvrop @ V)
#Hnm = vecs.conj().T @ (Hdvrop @ vecs)
#print(Hnm)


exit()


hij = model.hij(R = R_sub)
Vik = model.Vik()
Cij0 = np.zeros((model.ndvr, model.ndvr), dtype=np.complex128)
Cij0[model.map_ij, model.map_kl] = vecs[:,0]

Cij0_cp = Cij0.copy()
mat1 = np.matmul(hij, Cij0_cp)
CI1 = mat1[model.map_ij, model.map_kl]
mata = Vik * Cij0_cp
CIa = mata[model.map_ij, model.map_kl]

Cij0[model.map_kl, model.map_ij] = -vecs[:,0]

mat2 = np.matmul(hij, Cij0)
CI2 = mat2[model.map_ij, model.map_kl]
matb = Vik * Cij0
CIb = matb[model.map_ij, model.map_kl]
print(np.max(np.absolute(matb - mata)))
print(np.max(np.absolute(mat2 - mat1)))
#print(np.max(np.absolute(Cij0 - Cij0_cp)))
print("Now CI: \n")
print(np.max(np.absolute(CI2 - CI1)))
print(np.max(np.absolute(CIb - CIa)))


