import numpy as np

RI = np.loadtxt("RI/RI.dat", dtype=np.float64)
DR = (RI[-1] - RI[0]) / (RI.shape[0] - 1)

I=5
mospec_Rm1 = np.load(f"RI/R{I-1}/mospec.npz")
mospec_R = np.load(f"RI/R{I}/mospec.npz")
mospec_Rp1 = np.load(f"RI/R{I+1}/mospec.npz")

ep_Rm1 = mospec_Rm1['ep']
ep_R = mospec_R['ep']
ep_Rp1 = mospec_Rp1['ep']
nmo = ep_R.shape[0]

cip_Rm1 = mospec_Rm1['cip']
cip_R = mospec_R['cip']
cip_Rp1 = mospec_Rp1['cip']

d1ep_R = mospec_R['d1ep']
d2ep_R = mospec_R['d2ep']
nac1_R = mospec_R['nac1']
nac2_R = mospec_R['nac2']

nac1_approx1 = (cip_R.conj().T @ cip_Rp1 - np.eye(nmo)) / DR
nac1_approx2 = (np.eye(nmo) - cip_R.conj().T @ cip_Rm1) / DR

d2ep_approx0 = (ep_Rp1 - 2 * ep_R + ep_Rm1) / (DR**2)
nac2_approx0 = (cip_R.conj().T @ cip_Rp1 - 2 * np.eye(nmo) + cip_R.conj().T @ cip_Rm1) / (DR**2)

p=22
q=23
print(f'd2ep: 0: {d2ep_approx0[p]}, res: {d2ep_R[p]}')
print(f'nac1: 1: {nac1_approx1[p,q]}, 2: {nac1_approx2[p,q]}, res: {nac1_R[p,q]}')
print(f'nac2: 0: {nac2_approx0[p,q]}, res: {nac2_R[p,q]}')
