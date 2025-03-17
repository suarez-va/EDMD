import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))

from colbert_miller_dvr import dvr_p, dvr_T

m = 1.7
a = -10.3
b = 10.7
N = 10000
pmat = dvr_p(a, b, N, "(-infty,infty)")
Tmat = dvr_T(m, a, b, N, "(-infty,infty)")
pmat2 = np.einsum('ij,jk->ik', pmat, pmat)
Tmat2 = 2 * m * Tmat
for i in range(N):
    print(pmat2[i,i], Tmat2[i,i])

#print(np.matmul(pmat, pmat) / (2 * m))
#print(Tmat)
