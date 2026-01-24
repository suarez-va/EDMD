import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


timedata = np.load('timedata.npz')

t = timedata['t']
xi = timedata['xi']
rhot = timedata['rhot']

print(xi.shape)
print(xi[200:250])
print(xi[251:301])
tpts = t.shape[0]

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
ax.set_xlim(-100.0, 100.0)
ax.set_ylim(-0.001, 0.25)
#ax.axhline(y=0.0, linestyle='--', color='k')
ax.set_xlabel('x (a.u.)')
ax.set_ylabel('rho')
title = ax.set_title("")
plot, = ax.plot([],[], color='k')


P1t = np.sum(rhot[200:250,:],axis=0)
P2t = np.sum(rhot[251:301,:],axis=0)

def update(frame):
    title.set_text(f"t = {np.round(t[frame], 3)}; P1={np.round(P1t[frame], 3)}; P2={np.round(P2t[frame], 3)}; P12={np.round(P1t[frame] + P2t[frame], 3)}")
    plot.set_data(xi, rhot[:,frame])
    return [plot, title]

ani = FuncAnimation(fig, func=update, frames=tpts, blit=False)

plt.show()


