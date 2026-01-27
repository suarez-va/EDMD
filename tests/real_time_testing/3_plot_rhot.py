from datetime import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

#timedata = np.load('timedata.npz')
timedata = np.load('Lindblad.npz')

t = timedata['t']
xi = timedata['xi']
n1t = timedata['n1t']
n2t = timedata['n2t']
rhot = n1t + n2t
qt = 2.0 - np.sum(rhot, axis=0)

dx = (xi[-1] - xi[0]) / (xi.shape[0] - 1)
tpts = t.shape[0]

plt.rcParams.update({
    'figure.figsize': (6.0, 5.0),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 17,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xlim(-19.0, 19.0)
ax1.set_ylim(-0.001, 0.5)
ax1.set_xlabel('x (a.u.)', labelpad=-2)
ax1.set_ylabel('ρ(x)dx (a.u.)')
title = ax1.set_title("", fontsize=20)
plot1, = ax1.plot([],[], color='b', label='1-electron')
plot2, = ax1.plot([],[], color='r', label='2-electron')
plot3, = ax1.plot([],[], color='k', label='total')
ax1.legend(loc='upper right')

ax2.set_xlim(-1., 10000.)
ax2.set_ylim(-0.1, 2.1)
ax2.set_yticks([0, 1, 2])
ax2.set_xlabel('t (a.u.)', labelpad=-1)
ax2.set_ylabel('charge (a.u.)', labelpad=12)
ax2.plot(t, qt, color='k')

plot4 = ax2.axvline(x=0, color='k', linestyle='--')
plot5, = ax2.plot([],[], marker='o', color='k', markersize=7)
fig.subplots_adjust(hspace=0.3)

def update(frame):
    #title.set_text(f"t = {np.round(t[frame], 3)} (a.u.)")
    title.set_text(f"t = {int(t[frame])} (a.u.)")
    plot1.set_data(xi, n1t[:,frame] / dx)
    plot2.set_data(xi, n2t[:,frame] / dx)
    plot3.set_data(xi, rhot[:,frame] / dx)
    plot4.set_xdata([t[frame],t[frame]])
    plot5.set_data([t[frame]], [qt[frame]])
    return [plot1, plot2, plot3, plot4, plot5, title]

ani = FuncAnimation(fig, func=update, frames=tpts, blit=False)

ani.save("Lindblad.mp4", writer=FFMpegWriter(fps=240))

#plt.show()
