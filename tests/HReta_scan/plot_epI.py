import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if not os.path.exists("RI"):
    print("Missing grid data directory RI")
    exit()
mospec_0 = np.load("RI/R0/mospec.npz")
RI = np.loadtxt("RI/RI.dat", dtype=np.float64)

nmo = mospec_0["ep"].shape[0]
Ipts = RI.shape[0]
epI = np.zeros((nmo,Ipts), dtype=np.float64)
nac = np.zeros((Ipts), dtype=np.float64)
for I, R in enumerate(RI):
    sub_dir = f"RI/R{I}"
    mospec = np.load(sub_dir + "/mospec.npz")
    epI[:,I] = mospec["ep"]
    nac[I] = mospec["nac1"][22,23]

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

fig, ax = plt.subplots()
#ax.set_xlim(-1.0, 0.0)
ax.set_ylim(-1.5, 0.1)
ax.set_xlabel('R (a.u.)')
ax.set_ylabel('ε(R) (a.u.)')
for p in range(nmo):
    ax.plot(RI, epI[p,:])

#plt.show()

plt.savefig("epI.png")

