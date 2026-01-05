import numpy as np
from models.two_electron_gaussian_diatomic import TEGD, TESD
import matplotlib.pyplot as plt

paramsSD = {
    "aee": 0.02,
    "bee": 0.25,
    "aR": 0.01205,
    "bR": 0.01,
    "aAe": 0.0102,
    "bAe": 0.655,
    "aBe": 0.0139,
    "bBe": 0.473,
    "mA": 36443.98900696,
    "mB": 7294.29954142,
}

paramsGD = {
    "aR": 0.01205,
    "bR": 0.01,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.01, #0.1,
    "bee": 0.0, #100.0,
}

#model = TESD(a=-30.0, b=30.0, N=100, bounds="(-inf,inf)", spin="triplet",model_params=paramsSD)
model = TEGD(a=-35.0, b=35.0, N=115, bounds="(-inf,inf)", spin="singlet",model_params=paramsGD)

nsta = 111
Rpts = 55
Rmin = 0.0
Rmax = 20
R_ar = np.linspace(Rmin, Rmax, Rpts)

E_ar = np.zeros((Rpts, nsta))
for i, R in enumerate(R_ar):
    #En, Cijn = model.solve_wfn(R, nbo = nsta + 1)
    En, Cijn = model.solve_wfn(R)
    E_ar[i,:] = En[:nsta] + 0.0 * model.VR(R)
    print(i)

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
xi = model.xi()
for i in range(nsta):
    Pi = 2 * np.sum(np.absolute(Cijn[:,:,0])**2, axis=1)
    plt.plot(xi, Pi)
plt.savefig("rho.png")

plt.figure()
plt.title(f"interacting PES")
plt.xlabel("R (a.u.)")
plt.ylabel("En(R) (a.u.)")
for i in range(nsta):
    plt.plot(R_ar, E_ar[:,i], linewidth='2')
#plt.xlim([-12.0,12.0])
#plt.ylim([-2.0,0.5])
#plt.subplots_adjust(hspace=0.05, left=0.23, right=0.98, top=0.98, bottom=0.12)
plt.subplots_adjust(hspace=0.05, left=0.23, right=0.95, top=0.90, bottom=0.20)
plt.savefig("PES.png")



