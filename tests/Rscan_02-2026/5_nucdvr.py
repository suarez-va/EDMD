from time_independent.grid_utils import MolecularHamiltonian, NUCDVR, FCINUCDVR
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

M = 1836.0
K = 500

nucdvr = NUCDVR(nucgrid_file='RI/fcidvr_1ele_doublet_nucgrid.npz', mass=M)
print(np.min(nucdvr.data['EnI'][0,:]))
print(np.min(nucdvr.data['EnI'][1,:]))
print(np.min(nucdvr.data['EnI'][2,:]))

#H = nucdvr.HamiltonianOperator()
H = MolecularHamiltonian(nucgrid_file='RI/fcidvr_1ele_doublet_nucgrid.npz', mass=M)
print(H.shape[0])
nbo = nucdvr.nbo

time1 = time.time()
EN, CN = eigsh(H, k = K, which = 'SA')
time2 = time.time()
print(f'time = {(time2 - time1)/60.0} min')
idx = np.argsort(EN)
EN = EN[idx]; CN = CN[:,idx]

print(EN)

from model_systems.models import GICD, BHAR

params = {
    'ZA': 0.5,
    'ZB': 0.5,
    'aR': 0.0,
    'bR': 0.0001,
    'DA' : 1.0,
    'bA': 0.25,
    'DB' : 0.8,
    'bB': 1.0,
    'aee': 0.0,
    'bee': 0.0001,
}

params2 = {
    'ZA': 1.0,
    'ZB': 1.0,
    'aR': 0.0,
    'bR': 0.0001,
    'k': 0.05,
    'mu_mA': 0.9,
    'aee': 0.0,
    'bee': 0.0001,
}

Ra = 0.0
Rb = 10.0
RN = 250

model = GICD(params)
#model = BHAR(params2)
fcinucdvr = FCINUCDVR(model = model, xa = -196.7/2.0, xb = 196.7/2.0, xN = 250, xbounds = "(-inf,inf)", mass=M, Ra=Ra, Rb=Rb, RN=RN)
#fcinucdvr = FCINUCDVR(model = model, xa = -25, xb = 25, xN = 100, xbounds = "(-inf,inf)", mass=M, Ra=Ra, Rb=Rb, RN=RN)
Hfci = fcinucdvr.H_NUC(nele=1, spin='doublet')
print(Hfci.shape[0])
nxdvr = fcinucdvr.nxdvr
nRdvr = fcinucdvr.nRdvr

time1 = time.time()
En, Cn = eigsh(Hfci, k = K, which = 'SA')
time2 = time.time()
print(f'time = {(time2 - time1)/60.0} min')
idx = np.argsort(En)
En = En[idx]; Cn = Cn[:,idx]

Error = (EN - En)
print(Error)
imax = int(np.argmax(np.absolute(Error)))
print((imax, Error[imax], EN[imax], En[imax]))

plt.rcParams.update({
    'figure.figsize': (6.0, 7.0),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})
CnIN = CN.reshape(nbo, nRdvr, K)
nmax = np.argmax(np.einsum('nIN,nIN->nN', CnIN.conj(), CnIN), axis=0)
print(nmax)

nplt1 = 11
nplt2 = 12
print(nmax[nplt1],nmax[nplt2])
RI = fcinucdvr.RI()
CniI1 = Cn[:,nplt1].reshape(nxdvr, nRdvr)
CniI2 = Cn[:,nplt2].reshape(nxdvr, nRdvr)
CnI1 = np.einsum('Ii,iI->I', CniI1.conj().T, CniI1)
CnI2 = np.einsum('Ii,iI->I', CniI2.conj().T, CniI2)

CNnI1 = CN[:,nplt1].reshape(nbo, nRdvr)
CNnI2 = CN[:,nplt2].reshape(nbo, nRdvr)
CNI1 = np.einsum('In,nI->I', CNnI1.conj().T, CNnI1)
CNI2 = np.einsum('In,nI->I', CNnI2.conj().T, CNnI2)

fig, (ax1, ax2) = plt.subplots(2,1)
#ax1.set_xlim(0.0, 5.0)
#ax1.set_ylim(-1.0, 1.0)
ax1.set_xlabel('R (a.u.)')
ax1.set_ylabel('CiI (a.u.)')

ax2.set_ylabel('En (a.u.)')
ax2.set_ylabel('EN (a.u.)')

ax1.plot(RI, CnI1.real, color='k')
ax1.plot(RI, CnI2.real, color='k')
ax1.plot(RI, CNI1.real, color='g', linestyle='--')
ax1.plot(RI, CNI2.real, color='g', linestyle='--')
for n in range(K):
    ax2.axhline(y=En[n], color='k', linewidth=1.0)
    ax2.axhline(y=EN[n], color='g', linewidth=1.0, linestyle='--')

plt.show()

#plt.savefig("EnR.png")
