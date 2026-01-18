import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if not os.path.exists("etal"):
    print("Missing grid data directory etal")
    exit()
eigspec = np.load("eigspec.npz")
etal = np.loadtxt("etal/etal.dat", dtype=np.float64)

nbo = eigspec["En"].shape[0]
lpts = etal.shape[0]
Enl = np.zeros((nbo,lpts), dtype=np.complex128)
for l, eta in enumerate(etal):
    sub_dir = f"etal/eta{l}"
    capspec = np.load(sub_dir + "/capspec.npz")
    Enl[:,l] = capspec["En"]

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
ax.set_xlim(-1.0, 0.0)
ax.set_ylim(-0.005, 0.001)
ax.axhline(y=0.0, linestyle='--', color='k')
ax.set_xlabel('Re[E(η)]')
ax.set_ylabel('Im[E(η)]')
title = ax.set_title("")
plot, = ax.plot([],[],'ro', markersize=5)
texts = [ax.text(0, 0, '', ha='center', va='center', fontsize=9) for _ in range(nbo)]

def update(frame):
    title.set_text(f"η = {np.round(etal[frame], 8)}")
    plot.set_data(Enl[:,frame].real, Enl[:,frame].imag)
    for n, text in enumerate(texts):
        text.set_position((Enl[n, frame].real, Enl[n, frame].imag))
        text.set_text(str(n))
    return [plot, title] + texts

ani = FuncAnimation(fig, func=update, frames=lpts, blit=False)

plt.show()

#fig, ax = plt.subplots()
#ax.set_xlim(-1.0, 0.0)
#ax.set_ylim(-0.005, 0.0)
#for l, eta in enumerate(etal):
#    sub_dir = f"etal/eta{l}"
#    capspec = np.load(sub_dir + "/capspec.npz")
#    En = capspec["En"] ; Cnml = capspec["Cnml"]; Cnmr = capspec["Cnmr"]
#    for n, E in enumerate(En):
#        plt.text(E.real, E.imag, str(n), ha='center', va='center')
#    plt.plot(En.real,En.imag,'ro')
##plt.xlim([0.0,10.0])
#plt.ylim([-0.005,0.0])
#plt.savefig("Eeta.png")
#exit()

