import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

timedata = np.load('Lindblad.npz')

t = timedata['t']
xi = timedata['xi']
Px1t = timedata['Px1t']
Px2t = timedata['Px2t']
ep = timedata['ep']
En = timedata['En']
PE1t = timedata['PE1t']
PE2t = timedata['PE2t']
rhot = Px1t + Px2t
qt = 2.0 - np.sum(rhot, axis=0)

dx = (xi[-1] - xi[0]) / (xi.shape[0] - 1)
tpts = t.shape[0]

plt.rcParams.update({
    'figure.figsize': (6.0, 8.0),
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

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.set_xlim(-19.0, 19.0)
ax1.set_ylim(-0.001, 0.5)
ax1.set_xlabel('x (a.u.)', labelpad=-2)
ax1.set_ylabel('ρ(x)dx (a.u.)')
title = ax1.set_title("", fontsize=20)
plot1, = ax1.plot([],[], color='b', label='1-electron')
plot2, = ax1.plot([],[], color='r', label='2-electron')
plot3, = ax1.plot([],[], color='k', label='total')
ax1.legend(loc='upper right')

ax2.set_xlim(-0.92, 0.07)
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel('E$_i$ (a.u.)', labelpad=-1)
ax2.set_ylabel('P(E$_i$)', labelpad=12)
plot4 = ax2.vlines([],[],[], color='b', label='1-electron')
plot5 = ax2.vlines([],[],[], color='r', label='2-electron')
ax2.legend(loc='upper right')

ax3.set_xlim(-1., 10000.)
ax3.set_ylim(-0.1, 2.1)
ax3.set_yticks([0, 1, 2])
ax3.set_xlabel('t (a.u.)', labelpad=-1)
ax3.set_ylabel('charge (a.u.)', labelpad=12)
ax3.plot(t, qt, color='k')

plot6 = ax3.axvline(x=0, color='k', linestyle='--')
plot7, = ax3.plot([],[], marker='o', color='k', markersize=7)
fig.subplots_adjust(hspace=0.3)

def update(frame):
    #title.set_text(f"t = {np.round(t[frame], 3)} (a.u.)")
    title.set_text(f"t = {int(t[frame])} (a.u.)")
    plot1.set_data(xi, Px1t[:,frame] / dx)
    plot2.set_data(xi, Px2t[:,frame] / dx)
    plot3.set_data(xi, rhot[:,frame] / dx)
    plot4.set_segments([[(x,0),(x,y)] for x,y in zip(ep,PE1t[:,frame])])
    plot5.set_segments([[(x,0),(x,y)] for x,y in zip(En,PE2t[:,frame])])
    plot6.set_xdata([t[frame], t[frame]])
    plot7.set_data([t[frame]], [qt[frame]])
    return [plot1, plot2, plot3, plot4, plot5, plot6, plot7, title]

ani = FuncAnimation(fig, func=update, frames=tpts, blit=False)

ani.save("Lindblad.mp4", writer=FFMpegWriter(fps=240))

#plt.show()
