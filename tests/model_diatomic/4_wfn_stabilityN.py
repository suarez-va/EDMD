import numpy as np
from models.two_electron_gaussian_diatomic import TEGD
import matplotlib.pyplot as plt

params = {
    "aee": 0.1,
    "bee": 100.0,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
}

R = 8.0
ab = 40.0
nsta = 100
Npts = 26
N_ar = 20 + 4 * np.arange(Npts)
print(N_ar[0], N_ar[-1])

En_ar = np.zeros((Npts,nsta))

for i, N in enumerate(N_ar):
    model = TEGD(a=-ab, b=ab, N=N, bounds="(-inf,inf)", spin="tripet",model_params=params)
    En, Cijn = model.solve_wfn(R, nbo = nsta + 1)
    En_ar[i,:] = En[:nsta]

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
plt.title(f"Hdvr En; (a,b)=(-{ab},{ab}), R = {R} a.u.")
plt.xlabel("N points")
plt.ylabel("En (a.u.)")
for i in range(nsta):
    plt.plot(N_ar, En_ar[:,i], linewidth='1.2')
#plt.xlim([-12.0,12.0])
#plt.ylim([-0.1,0.0])
#plt.subplots_adjust(hspace=0.05, left=0.23, right=0.98, top=0.98, bottom=0.12)
#plt.subplots_adjust(hspace=0.05, left=0.23, right=0.95, top=0.90, bottom=0.20)
plt.savefig("HdvrstN.png")

