import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if not os.path.exists("etai"):
    print("Missing grid data directory etai")
    exit()

capgrid_data = np.load('etai/fcicap_2ele_singlet_capgrid.npz')
etai = capgrid_data['etai']
Eni = capgrid_data['Eni']
ipts = etai.shape[0]
nbo = Eni.shape[0]


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
    title.set_text(f"η = {np.round(etai[frame], 8)}")
    plot.set_data(Eni[:,frame].real, Eni[:,frame].imag)
    for n, text in enumerate(texts):
        text.set_position((Eni[n, frame].real, Eni[n, frame].imag))
        text.set_text(str(n))
    return [plot, title] + texts

ani = FuncAnimation(fig, func=update, frames=ipts, blit=False)

plt.show()

