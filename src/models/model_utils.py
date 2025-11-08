import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))
from grid_utils.colbert_miller_dvr import dvr_xn, dvr_T, dvr_W

def calculate_nac(E, dHele_adi):
    nstates = E.shape[0]
    nac = np.zeros((nstates, nstates), dtype=complex)
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                pass
            else:
                nac[i,j] = dHele_adi[i,j] / (E[j] - E[i])
    return nac

