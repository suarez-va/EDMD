import numpy as np
from models.two_electron_gaussian_diatomic import TEGD, TESD
import matplotlib.pyplot as plt


#paramsSD = {
#    "ZA": 1,
#    "ZB": 1,
#    "aee": 0.02,
#    "bee": 0.25,
#    "aR": 0.01205,
#    "bR": 0.01,
#    "aAe": 0.0102,
#    "bAe": 0.655,
#    "aBe": 0.0139,
#    "bBe": 0.473,
#    "mA": 36443.98900696,
#    "mB": 7294.29954142,
#}

paramsSD = {
    "ZA": 2,
    "ZB": 1,
    "mA": 7294.0,
    "mB": 1836.0,
    "aR": 0.01205,
    "bR": 0.01,
    "aAe": 0.002,
    "bAe": 1.0,
    "aBe": 0.002,
    "bBe": 1.0,
    "aee": 0.02,
    "bee": 0.25,
}

paramsGD = {
    "aR": 0.01205,
    "bR": 0.01,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.1,
    "bee": 100.0,
}

model = TESD(a=-100.0, b=100.0, N=500, bounds="(-inf,inf)", spin="triplet",model_params=paramsSD)
#model = TEGD(a=-100.0, b=100.0, N=350, bounds="(-inf,inf)", spin="triplet",model_params=paramsGD)

nsta = 20
Rpts = 51
Rmin = 0.0
Rmax = 20
R_ar = np.linspace(Rmin, Rmax, Rpts)

E_ar = np.zeros((Rpts, nsta))
for i, R in enumerate(R_ar):
    ep, cip = model.solve_mos(R)
    E_ar[i,:] = ep[:nsta] + 0.0 * model.VR(R)
    #print(i)
    #E_ar[i,:] = model.VR(R) - 2.0

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
plt.title(f"non-interacting PES")
plt.xlabel("R (a.u.)")
plt.ylabel("En(R) (a.u.)")
for i in range(nsta):
    plt.plot(R_ar, E_ar[:,i], linewidth='2')
#plt.xlim([-12.0,12.0])
plt.ylim([-2.7,0.5])
#plt.subplots_adjust(hspace=0.05, left=0.23, right=0.98, top=0.98, bottom=0.12)
plt.subplots_adjust(hspace=0.05, left=0.23, right=0.95, top=0.90, bottom=0.20)
plt.savefig("pes.png")

