import numpy as np
from model_systems.models import GICD
from time_independent.fcidvr import validate_nele_spin, fci_mapping, fci_operator, fci_wfn, fci_1rdm, fci_dyson, FCIDVR, compute_cap, compute_1rdm, compute_rho, compute_dyson
import matplotlib.pyplot as plt
import time


params = {
    'ZA': 0.0,
    'ZB': 0.0,
    'aR': 0.0,
    'bR': 0.0001,
    'DA' : 1.0,
    'bA': 0.25,
    'DB' : 0.8,
    'bB': 1.0,
    'aee': 0.0,
    'bee': 0.0001,
}

R_sub = 8.0
nbo_sub1 = 25
nbo_sub2 = 7
nbo_sub3 = 5

model = GICD(params)
fcidvr = FCIDVR(model = model, xa = -196.7/2.0, xb = 196.7/2.0, xN = 541, xbounds = "(-inf,inf)")

ndvr = fcidvr.nxdvr
hij = fcidvr.hij(R_sub)
gik = fcidvr.gik()
nfci_1ele, map_norm_1ele, map_idx_1ele = fci_mapping(ndvr = ndvr, nele = 1, spin = 'doublet')
#nfci_2ele, map_norm_2ele, map_idx_2ele = fci_mapping(ndvr = ndvr, nele = 2, spin = 'singlet')
nfci_2ele, map_norm_2ele, map_idx_2ele = fci_mapping(ndvr = ndvr, nele = 2, spin = 'triplet')
print(nfci_1ele)
print(nfci_2ele)

#time1=time.time()
#fcidvr.kernel(R = R_sub, nele = 1, spin = 'doublet', nbo = nbo_sub1, derivative_order = 0)
#time2=time.time()
#print(f'time: {(time2 - time1)/60.0} min')
#time1=time.time()
##fcidvr.kernel(R = R_sub, nele = 2, spin = 'singlet', nbo = nbo_sub2, derivative_order = 0)
#fcidvr.kernel(R = R_sub, nele = 2, spin = 'triplet', nbo = nbo_sub2, derivative_order = 0)
#time2=time.time()
#print(f'time: {(time2 - time1)/60.0} min')

#fcidvr_data1 = np.load('fcidvr_1ele_doublet.npz')
#fcidvr_data2 = np.load('fcidvr_2ele_singlet.npz')
#fcidvr_data2 = np.load('fcidvr_2ele_triplet.npz')

H1 = fci_operator(ndvr, 1, 'doublet', hij, gik)
#H2 = fci_operator(ndvr, 2, 'singlet', hij, gik)
H2 = fci_operator(ndvr, 2, 'triplet', hij, gik)

C2test = np.random.random(nfci_2ele) + 1j*np.random.random(nfci_2ele)
C2test *= 1/np.sqrt(np.sum(C2test.conj()*C2test))

import cProfile

profiler = cProfile.Profile()
profiler.enable()

for _ in range(1000):
    H2.matvec(C2test)

profiler.disable()
profiler.print_stats(sort="cumtime")

