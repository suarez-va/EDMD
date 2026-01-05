from grid_utils.generate_grid import sort_eta_data, get_eta_data
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (4.0, 3.0),
    'figure.dpi': 200,
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

#eta_ar, Er_ar, Gam_ar = get_eta_data(11) # ab=30 11
#eta_ar, Er_ar, Gam_ar = get_eta_data(13) # ab=35 13
#eta_ar, Er_ar, Gam_ar = get_eta_data(16) # ab=40 16 R=8
eta_ar, Er_ar, Gam_ar = get_eta_data(11) # ab=40 11 R=8.5

#plt.figure()
#npts = eta_ar.shape[0]
#for i in range(npts):
#    plt.text(Er_ar[i], Gam_ar[i], str(i), ha='center', va='center')
#plt.plot(Er_ar[:npts],Gam_ar[:npts],'ro')
##plt.xlim([0.0,10.0])
##plt.ylim([-0.1,0.0])
##plt.ylim([-0.03,0.0])
#plt.savefig("traj.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

npts = eta_ar.shape[0]
for i in range(npts):
    ax1.text(Er_ar[i], Gam_ar[i], str(i), ha='center', va='center')
ax1.plot(Er_ar[:npts],Gam_ar[:npts],'ro')
ax1.set_xlabel('Re[E(η)]')
ax1.set_ylabel('Im[E(η)]')

E_ar = Er_ar + 1j * Gam_ar

dE_ar = np.zeros(eta_ar.shape[0]-1, dtype=np.complex128)
for i in range(dE_ar.shape[0]):
    dE_ar[i] = (E_ar[i+1]-E_ar[i])/(eta_ar[i+1]-eta_ar[i])

etadE_ar = np.absolute(eta_ar[:-1] * dE_ar)
#ax2.plot(eta_ar[:-1], etadE_ar, color='k')
ax2.plot(np.arange(etadE_ar.shape[0]), etadE_ar, color='k')
ax2.set_xlabel('η index')
ax2.set_ylabel('η|dE(η)/dη|')
plt.subplots_adjust(hspace=0.2, left=0.23, right=0.95, top=0.90, bottom=0.20)
plt.savefig("traj.png")

