import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from grid_utils.colbert_miller_dvr import dvr_p, dvr_T, dvr_xn, dvr_W

m = 1.0
ab = 2.0
N = 750

xmat = dvr_xn(1, -ab, ab, N, "(-inf,inf)")
xi=np.diagonal(xmat).real
Tmat = dvr_T(m, -ab, ab, N, "(-inf,inf)")

bcap = 1
wi = (xi**2 * (np.heaviside(xi - 0, 0.5) - np.heaviside(xi - bcap, 0.5)))
Wmat = np.diag(wi)

plt.figure()
plt.plot(xi,wi)
plt.savefig("wi.png")


eta = 250.0
H = Tmat - 1j * eta * Wmat
Eraw, Craw = np.linalg.eig(H)
idx = np.argsort(Eraw.real)
E = Eraw[idx]
C = Craw[:, idx]

print(E.imag)

plt.rcParams.update({
    'figure.figsize': (4.0, 3.0),
    'figure.dpi': 300,
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

plt.figure()
plt.axvline(x=0, linestyle='--', color='k')
plt.axvline(x=bcap, linestyle='--', color='k')
plt.plot(xi,np.abs(C[:,4])**2, color='b')
plt.plot(xi,np.abs(C[:,3])**2, color='g')
plt.plot(xi,np.abs(C[:,2])**2, color='gold')
plt.plot(xi,np.abs(C[:,1])**2, color='orange')
plt.plot(xi,np.abs(C[:,0])**2, color='r')
#plt.plot(xi,np.abs(C[:,5])**2, color='darkviolet')
plt.xlim([-ab,ab])
plt.savefig("fpwfn.png")

