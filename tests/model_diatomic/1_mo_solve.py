import numpy as np
from models.two_electron_screened_diatomic import TESD
import matplotlib.pyplot as plt

params = {
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
    "xmax": 40.0,
    "ndvr": 500,
    "spin": "singlet",
    "nbo": 250
}
params["aAe"] = 0.10 #0.25
params["aBe"] = 0.50 #1.00
params["bAe"] = 1.00
params["bBe"] = 1.5625

R = 8.0
model = TESD(params)
model.solve_mos(R)

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


npts = 10
nscale = 2.2
xi = model.xi()
Vi = model.VeR(xi, R)
ep = model.ep.real
cip = model.cip
plt.figure()
plt.title(f"hcore spectra; R = {R} a.u.")
plt.xlabel("x (a.u.)")
plt.ylabel("V(x;R) (a.u.)")
plt.plot(xi, Vi, color='k', linewidth='2')
for i in range(npts):
    plt.plot(xi, nscale * np.absolute(cip[:,i])**2 + ep[i], linewidth='1')
    plt.axhline(y=ep[i], color='k', linewidth='1', linestyle='--')
plt.xlim([-12.0,12.0])
#plt.ylim([-0.1,0.0])
#plt.subplots_adjust(hspace=0.05, left=0.23, right=0.98, top=0.98, bottom=0.12)
plt.subplots_adjust(hspace=0.05, left=0.23, right=0.95, top=0.90, bottom=0.20)
plt.savefig("mospec.png")

