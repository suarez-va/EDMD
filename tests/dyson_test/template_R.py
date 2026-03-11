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
nbo_sub2 = 75

model = GICD(params)
fcidvr = FCIDVR(model = model, xa = -196.7/2.0, xb = 196.7/2.0, xN = 375, xbounds = "(-inf,inf)")

ndvr = fcidvr.nxdvr
nfci_doublet, map_norm_doublet, map_idx_doublet = fci_mapping(ndvr = ndvr, nele = 1, spin = 'doublet')
nfci_singlet, map_norm_singlet, map_idx_singlet = fci_mapping(ndvr = ndvr, nele = 2, spin = 'singlet')

time1=time.time()
fcidvr.kernel(R = R_sub, nele = 1, spin = 'doublet', nbo = nbo_sub1, derivative_order = 0)
fcidvr.kernel(R = R_sub, nele = 2, spin = 'singlet', nbo = nbo_sub2, derivative_order = 0)
time2=time.time()
print(f'time: {(time2 - time1)/60.0} min')

time1=time.time()
compute_rho('fcidvr_1ele_doublet.npz', 0.5)
time2=time.time()
print(f'time: {(time2 - time1)/60.0} min')

time1=time.time()
compute_rho('fcidvr_1ele_doublet.npz', -0.5)
time2=time.time()
print(f'time: {(time2 - time1)/60.0} min')

time1=time.time()
compute_rho('fcidvr_2ele_singlet.npz', 0.0)
time2=time.time()
print(f'time: {(time2 - time1)/60.0} min')

time1=time.time()
compute_dyson('fcidvr_1ele_doublet.npz', 0.5, 'fcidvr_2ele_singlet.npz', 0.0)
time2=time.time()
print(f'time: {(time2 - time1)/60.0} min')

time1=time.time()
compute_dyson('fcidvr_1ele_doublet.npz', -0.5, 'fcidvr_2ele_singlet.npz', 0.0)
time2=time.time()
print(f'time: {(time2 - time1)/60.0} min')

xi = fcidvr.xi()
#Dibpn = np.load('fcidvr_1ele_doublet_0.5Sz_2ele_singlet_0.0Sz_dyson.npz')['Dispn']
#Diapn = np.load('fcidvr_1ele_doublet_-0.5Sz_2ele_singlet_0.0Sz_dyson.npz')['Dispn']
#print(Dibpn[:,1,0,0])
#print(Diapn[:,0,0,0])

plt.rcParams.update({
    'figure.figsize': (6.0, 4.0),
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

p = 0
n = 43

fcidvr_data1 = np.load('fcidvr_1ele_doublet.npz')
fcidvr_data2 = np.load('fcidvr_2ele_singlet.npz')
gam1 = fci_1rdm(ndvr, 1, 'doublet', 0.5, fcidvr_data1['Cn'][:,p])
gam2 = fci_1rdm(ndvr, 2, 'singlet', 0.0, fcidvr_data2['Cn'][:,n])
rho1 = np.diag(gam1[:,0,:,0].real + gam1[:,1,:,1].real)
rho2 = np.diag(gam2[:,0,:,0].real + gam2[:,1,:,1].real)
#print(np.sum(rho1))
#print(np.sum(rho2))
Dia = fci_dyson(ndvr, 1, 'doublet', -0.5, fcidvr_data1['Cn'][:,p], 2, 'singlet', 0.0, fcidvr_data2['Cn'][:,n])
Di = Dia[:,0]
#Di=Diapn[:,0,p,n]

fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4))
ax1.set_xlabel('D_pn(x)')
ax1.set_ylabel('x')
ax1.set_title(f"p = {p}; n = {n}")
ax1.set_xlim([-35.0, 35.0])
#ax1.set_ylim([-0.01, 0.2])
ax1.plot(xi, np.abs(Di), color='k')
#ax1.plot(xi, np.angle(Di), color='g')
ax1.plot(xi, rho1, color='r')
ax1.plot(xi, rho2, color='b', linestyle='--')
plt.show()
