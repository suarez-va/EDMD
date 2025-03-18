import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))

from colbert_miller_dvr import dvr_p, dvr_T

m = 1.7
a = 0.
b = 100.7
N = 2500
pmat = dvr_p(a, b, N, "(-inf,inf)")
Tmat = dvr_T(m, a, b, N, "(-inf,inf)")
pmat2 = np.einsum('ij,jk->ik', pmat, pmat)
Tmat2 = 2 * m * Tmat
for i in range(N):
    print(pmat2[i,i], Tmat2[i,i])

print("test2")
Tmat3 = dvr_T(m, a, b, N, None)
for i in range(N-2):
    print(Tmat[i+1,i+1] , Tmat3[i,i])

print("test3")
pmat3 = dvr_p(a, b, N, None)
for i in range(N-2):
    print(pmat[i+1,i+1] , pmat3[i,i])

print("test4")
Tmat_approx = np.matmul(pmat3, pmat3) / (2 * m)
for i in range(N-2):
    print(Tmat3[i,i], Tmat_approx[i,i])

print("test5")
pmat5 = dvr_p(a, b, N, "(0,inf)")
Tmat5_approx = np.matmul(pmat5, pmat5) / (2 * m)
Tmat5 = dvr_T(m, a, b, N, "(0,inf)")
for i in range(N-1):
    print(Tmat5[i,i], Tmat5_approx[i,i])




#print(np.matmul(pmat, pmat) / (2 * m))
#print(Tmat)
