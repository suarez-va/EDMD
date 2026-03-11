import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

output_file = 'RI/fcidvr_1ele_doublet_nucgrid.npz'
#output_file = 'RI/fcidvr_2ele_singlet_nucgrid.npz'
nucgrid = np.load(output_file)

RI = nucgrid['RI']
EnI = nucgrid['EnI']
d1EnI = nucgrid['d1EnI']
nac01I = nucgrid['nac01I']
nac11I = nucgrid['nac11I']

#print(nac1I.shape)
#print(np.unravel_index(np.argmax(np.absolute(nac1I)), nac1I.shape))
#print(nac1I[10,11,5])
#print(nac1I[11,12,5])
#print(nac1I[10,12,5])

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

nmax = 100

fig, (ax1, ax2) = plt.subplots(2,1)
#ax1.set_xlim(0.0, 5.0)
ax1.set_ylim(0.0, 10.0)
ax1.set_xlabel('R (a.u.)')
ax1.set_ylabel('E(R) (a.u.)')

for n in range(nmax):
    ax1.plot(RI, EnI[n,:])
    ax2.plot(RI, nac01I[n,n+1,:].real)

plt.show()

#plt.savefig("EnR.png")

