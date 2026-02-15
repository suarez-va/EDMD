import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if not os.path.exists("Rk"):
    print("Missing grid data directory Rk")
    exit()
eigspec_0 = np.load("Rk/R0/eigspec.npz")
Rk = np.loadtxt("Rk/Rk.dat", dtype=np.float64)

nbo = eigspec_0["En"].shape[0]
kpts = Rk.shape[0]
Enk = np.zeros((nbo,kpts), dtype=np.float64)
for k, R in enumerate(Rk):
    sub_dir = f"Rk/R{k}"
    eigspec = np.load(sub_dir + "/eigspec.npz")
    Enk[:,k] = eigspec["En"]

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
ax.set_xlim(4.0, 20.0)
ax.set_ylim(-0.5, -0.325)
ax.set_xlabel('R (a.u.)')
ax.set_ylabel('E(R) (a.u.)')
#ax.axvline(x=8.0, color='k', linestyle='--')
for n in range(250):
    if n==233:
        ax.plot(Rk, Enk[233,:], color = 'k', linewidth=2)
    else:
        ax.plot(Rk, Enk[n,:])
    #ax.plot(Rk, Enk[n,:] + 0.6936231602269626)

#plt.show()

plt.savefig("Enk_zoom.png")

