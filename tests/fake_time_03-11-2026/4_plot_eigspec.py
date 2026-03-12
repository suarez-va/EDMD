import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from time_independent.fcidvr import fci_wfn, fci_density

Ldata = np.load('Ldata.npz')

Ln  = Ldata['Ln']; Lpts = Ln.shape[0] 
epn = Ldata['epn']; nbo1 = epn.shape[0]
Enn = Ldata['Enn']; nbo2 = Enn.shape[0]
i00 = Ldata['i00']
i01 = Ldata['i01']
i11 = Ldata['i11']
i12 = Ldata['i12']
i22 = Ldata['i22']

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
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

fig, (ax) = plt.subplots(1, 1)
ax.set_xlim(0.0, 1.0)
#ax.set_ylim(-0.001, 0.5)
ax.set_xticks([0.2, 0.8])
ax.set_xticklabels(['2 electron', '1 electron'], fontsize=18)
ax.set_ylabel('En (a.u.)')

title = ax.set_title("", fontsize=20)

plot0 = ax.axhline(y=epn[0,0], xmin=0.65, xmax=0.95, color='r', linewidth=1, label='|0⟩')
plot1 = ax.axhline(y=epn[1,0], xmin=0.65, xmax=0.95, color='g', linewidth=1, label='|1⟩')
plot2 = ax.axhline(y=epn[2,0], xmin=0.65, xmax=0.95, color='b', linewidth=1, label='|2⟩')

plot01 = ax.axhline(y=Enn[i01[0],0], xmin=0.65, xmax=0.95, color='r', linewidth=1, label='|01⟩')
plot12 = ax.axhline(y=Enn[i12[0],0], xmin=0.65, xmax=0.95, color='b', linewidth=1, label='|21⟩')


#for p in range(nbo1):
#    if p==0:
#        ax.axhline(y=ep[0], xmin=0.65, xmax=0.95, color='r', linewidth=1, label='|0⟩')
#        ax.text(x=0.61, y=ep[0], s='|0⟩', ha='center', va='center', color='k', fontsize=14)
#    elif p==1:
#        ax.axhline(y=ep[1], xmin=0.65, xmax=0.95, color='g', linewidth=1, label='|1⟩')
#        ax.text(x=0.61, y=ep[1], s='|1⟩', ha='center', va='center', color='k', fontsize=14)
#    elif p==2:
#        ax.axhline(y=ep[2], xmin=0.65, xmax=0.95, color='b', linewidth=1, label='|2⟩')
#        ax.text(x=0.61, y=ep[2], s='|2⟩', ha='center', va='center', color='k', fontsize=14)
#    else:
#        ax.axhline(y=ep[p], xmin=0.65, xmax=0.95, color='k', linewidth=1)
#for n in range(nbo2):
#    if n == i01:
#        ax.axhline(y=En[i01], xmin=0.05, xmax=0.35, color='r', linewidth=1, label='|01⟩')
#        ax.text(x=0.40, y=En[i01], s='|01⟩', ha='center', va='center', color='k', fontsize=14)
#    elif n == i12:
#        ax.axhline(y=En[i12], xmin=0.05, xmax=0.35, color='b', linewidth=1, label='|21⟩')
#        ax.text(x=0.40, y=En[i12], s='|21⟩', ha='center', va='center', color='k', fontsize=14)
#    else:
#        ax.axhline(y=En[n], xmin=0.05, xmax=0.35, color='k', linewidth=1)

fig.subplots_adjust(hspace=0.25, left=0.17, right=0.95, top=0.95, bottom=0.06)

def update(frame):
    title.set_title("L = {int(Ln[frame])} (a.u.); R = 8 (a.u.)", fontsize=20)
    plot0.set_data(y=epn[0,frame])
    plot1.set_data(y=epn[1,frame])
    plot2.set_data(y=epn[2,frame])
    plot01.set_data(y=Enn[i01[frame],frame])
    plot12.set_data(y=Enn[i12[frame],frame])
    #plot4.set_segments([[(x,0),(x,y)] for x,y in zip(ep,PE1t[:,frame])])
    return [plot0, plot1, plot2, plot01, plot12, title]

ani = FuncAnimation(fig, func=update, frames=Lpts, blit=False)

ani.save("eigenspectra.mp4", writer=FFMpegWriter(fps=60))

#plt.show()
