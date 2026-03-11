import numpy as np
import matplotlib.pyplot as plt
from time_independent.fcidvr import fci_wfn, fci_density
from model_systems.models import GICD

params = {
    'ZA': 0.0,
    'ZB': 0.0,
    'aR': 0.0,
    'bR': 0.0001,
    'DA' : 1.0,
    'bA': 0.25,
    'DB' : 0.8,
    'bB': 1.0,
    'aee': 0.0,
    'bee': 0.0001,
}

R = 8.0
model = GICD(params)

fcidvr_data1 = np.load('fcidvr_1ele_doublet.npz')
#fcidvr_data_cap1 = np.load('fcidvr_1ele_doublet_cap.npz')
#fcidvr_data_density1 = np.load('fcidvr_1ele_doublet_0.5Sz_density.npz')
fcidvr_data2 = np.load('fcidvr_2ele_singlet.npz')
#fcidvr_data_cap2 = np.load('fcidvr_2ele_singlet_cap.npz')
#fcidvr_data_density2 = np.load('fcidvr_2ele_singlet_0.0Sz_density.npz')
#fcidvr_data_dyson = np.load('fcidvr_1ele_doublet_0.5Sz_2ele_singlet_0.0Sz_dyson.npz')

xi = fcidvr_data1["xi"]
ep = fcidvr_data1["En"]
cp = fcidvr_data1["Cn"]
#wi = fcidvr_data_cap1["wi"]
#wpq = fcidvr_data_cap1["Wnm"]
#Pipq = fcidvr_data_density1["Pisnm"][:,0,:,:]
En = fcidvr_data2["En"]
Cn = fcidvr_data2["Cn"]
#Wnm = fcidvr_data_cap2["Wnm"]
#Pinm = 2*fcidvr_data_density2["Pisnm"][:,0,:,:]
#Dipn = fcidvr_data_dyson["Dispn"][:,1,:,:]

ndvr = xi.shape[0]
nbo1 = ep.shape[0]
nbo2 = En.shape[0]
phi0a = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = 0.5, C = cp[:,0])
phi0b = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = -0.5, C = cp[:,0])
phi1a = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = 0.5, C = cp[:,1])
phi1b = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = -0.5, C = cp[:,1])
phi2a = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = 0.5, C = cp[:,2])
phi2b = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = -0.5, C = cp[:,2])
det0a0b = 1.0/np.sqrt(2)*(phi0a[:,:,None,None]*phi0b[None,None,:,:] - phi0b[:,:,None,None]*phi0a[None,None,:,:])
det0a1b = 1.0/np.sqrt(2)*(phi0a[:,:,None,None]*phi1b[None,None,:,:] - phi1b[:,:,None,None]*phi0a[None,None,:,:])
det1a1b = 1.0/np.sqrt(2)*(phi1a[:,:,None,None]*phi1b[None,None,:,:] - phi1b[:,:,None,None]*phi1a[None,None,:,:])
det1a2b = 1.0/np.sqrt(2)*(phi1a[:,:,None,None]*phi2b[None,None,:,:] - phi2b[:,:,None,None]*phi1a[None,None,:,:])
det2a2b = 1.0/np.sqrt(2)*(phi2a[:,:,None,None]*phi2b[None,None,:,:] - phi2b[:,:,None,None]*phi2a[None,None,:,:])
S00 = 0.5*(det0a0b + det0a0b.swapaxes(0,2))
S01 = 1.0/np.sqrt(2)*(det0a1b + det0a1b.swapaxes(0,2))
S11 = 0.5*(det1a1b + det1a1b.swapaxes(0,2))
S12 = 1.0/np.sqrt(2)*(det1a2b + det1a2b.swapaxes(0,2))
S22 = 0.5*(det2a2b + det2a2b.swapaxes(0,2))

C00 = np.zeros((nbo2), dtype=np.complex128)
C01 = np.zeros((nbo2), dtype=np.complex128)
C11 = np.zeros((nbo2), dtype=np.complex128)
C12 = np.zeros((nbo2), dtype=np.complex128)
C22 = np.zeros((nbo2), dtype=np.complex128)

for n in range(nbo2):
    wfn = fci_wfn(ndvr = ndvr, nele = 2, spin = 'singlet', Sz = 0, C = Cn[:,n])
    C00[n] = np.sum(wfn.conj()*S00)
    C01[n] = np.sum(wfn.conj()*S01)
    C11[n] = np.sum(wfn.conj()*S11)
    C12[n] = np.sum(wfn.conj()*S12)
    C22[n] = np.sum(wfn.conj()*S22)

i00 = np.argmax(np.abs(C00))
i01 = np.argmax(np.abs(C01))
i11 = np.argmax(np.abs(C11))
i12 = np.argmax(np.abs(C12))
i22 = np.argmax(np.abs(C22))
print(f'S00: i={i00}, |C|={np.abs(C00[i00])}')
print(f'S01: i={i01}, |C|={np.abs(C01[i01])}')
print(f'S11: i={i11}, |C|={np.abs(C11[i11])}')
print(f'S12: i={i12}, |C|={np.abs(C12[i12])}')
print(f'S22: i={i22}, |C|={np.abs(C22[i22])}')

dx = (xi[-1] - xi[0]) / (xi.shape[0] - 1)
VeR = model.VeR(x=xi, R=R)
rho0 = fci_density(ndvr=ndvr, nele=1, spinbra='doublet', Szbra=0.5, Cbra=cp[:,0]).sum(axis=1).real/dx
rho1 = fci_density(ndvr=ndvr, nele=1, spinbra='doublet', Szbra=0.5, Cbra=cp[:,1]).sum(axis=1).real/dx
rho2 = fci_density(ndvr=ndvr, nele=1, spinbra='doublet', Szbra=0.5, Cbra=cp[:,2]).sum(axis=1).real/dx
rho01 = fci_density(ndvr=ndvr, nele=2, spinbra='singlet', Szbra=0.0, Cbra=Cn[:,i01]).sum(axis=1).real/dx
rho12 = fci_density(ndvr=ndvr, nele=2, spinbra='singlet', Szbra=0.0, Cbra=Cn[:,i12]).sum(axis=1).real/dx
rho12m1 = fci_density(ndvr=ndvr, nele=2, spinbra='singlet', Szbra=0.0, Cbra=Cn[:,i12-1]).sum(axis=1).real/dx
rho12p1 = fci_density(ndvr=ndvr, nele=2, spinbra='singlet', Szbra=0.0, Cbra=Cn[:,i12+1]).sum(axis=1).real/dx

plt.rcParams.update({
    'figure.figsize': (6.0, 8.0),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 20,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

fig, (ax) = plt.subplots(1, 1)
ax.set_xlim(0.0, 1.0)
#ax.set_ylim(-0.001, 0.5)
ax.set_xticks([0.2, 0.8])
ax.set_xticklabels(['2 electron', '1 electron'], fontsize=18)
ax.set_ylabel('En (a.u.)')
ax.set_title("L = 100 (a.u.); R = 8 (a.u.)", fontsize=20)
for p in range(nbo1):
    if p==0:
        ax.axhline(y=ep[0], xmin=0.65, xmax=0.95, color='r', linewidth=1, label='|0⟩')
        ax.text(x=0.61, y=ep[0], s='|0⟩', ha='center', va='center', color='k', fontsize=14)
    elif p==1:
        ax.axhline(y=ep[1], xmin=0.65, xmax=0.95, color='g', linewidth=1, label='|1⟩')
        ax.text(x=0.61, y=ep[1], s='|1⟩', ha='center', va='center', color='k', fontsize=14)
    elif p==2:
        ax.axhline(y=ep[2], xmin=0.65, xmax=0.95, color='b', linewidth=1, label='|2⟩')
        ax.text(x=0.61, y=ep[2], s='|2⟩', ha='center', va='center', color='k', fontsize=14)
    else:
        ax.axhline(y=ep[p], xmin=0.65, xmax=0.95, color='k', linewidth=1)
for n in range(nbo2):
    if n == i01:
        ax.axhline(y=En[i01], xmin=0.05, xmax=0.35, color='r', linewidth=1, label='|01⟩')
        ax.text(x=0.40, y=En[i01], s='|01⟩', ha='center', va='center', color='k', fontsize=14)
    elif n == i12:
        ax.axhline(y=En[i12], xmin=0.05, xmax=0.35, color='b', linewidth=1, label='|21⟩')
        ax.text(x=0.40, y=En[i12], s='|21⟩', ha='center', va='center', color='k', fontsize=14)
    else:
        ax.axhline(y=En[n], xmin=0.05, xmax=0.35, color='k', linewidth=1)

fig.subplots_adjust(hspace=0.25, left=0.17, right=0.95, top=0.95, bottom=0.06)

plt.savefig("energy.png")


plt.rcParams.update({'figure.figsize': (6.0, 6.0)})
scale1=1.0
scale2=1.0
scale3=0.05
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.set_xlim(-25.0, 25.0)
ax1.set_ylim(-1.1, 0.15)
ax1.set_yticks([-1.0, 0.0])
ax1.set_ylabel('Energy (a.u.)')
ax1.set_title("L = 100 (a.u.); R = 8 (a.u.)", fontsize=20)
ax1.plot(xi, VeR, color='k')
ax1.axhline(y=ep[0], color='k', linestyle='--')
ax1.axhline(y=ep[1], color='k', linestyle='--')
ax1.axhline(y=ep[2], color='k', linestyle='--')
ax1.plot(xi, scale1*rho0 + ep[0], color='r', label='|0⟩')
ax1.plot(xi, scale1*rho1 + ep[1], color='g', label='|1⟩')
ax1.plot(xi, scale1*rho2 + ep[2], color='b', label='|2⟩')
ax1.legend(ncol=2, fontsize=9, loc='best', handletextpad=0.2,  columnspacing=0.9)

ax2.set_ylim(-1.1, 0.15)
ax2.set_yticks([-1.0, 0.0])
ax2.set_xlabel('x (a.u.)', labelpad=-2)
ax2.set_ylabel('Energy (a.u.)')
ax2.plot(xi, VeR, color='k')
ax2.axhline(y=En[i01], color='k', linestyle='--')
ax2.axhline(y=En[i12], color='k', linestyle='--')
ax2.plot(xi, scale2*rho01 + En[i01], color='r', label='|01⟩')
ax2.plot(xi, scale2*rho12 + En[i12], color='b', label='|21⟩')
ax2.legend(ncol=2, fontsize=9, loc='best', handletextpad=0.2,  columnspacing=0.9)

#print(En[i12-1], En[i12+1])
#ax3.set_ylim(-0.45, -0.31)
#ax3.set_xlabel('x (a.u.)', labelpad=-2)
#ax3.set_ylabel('Energy (a.u.)')
#ax3.plot(xi, scale3*rho12m1 + En[i12-1], color='r', label='1 below |21⟩')
#ax3.plot(xi, scale3*rho12 + En[i12], color='k', label='ρ21')
#ax3.plot(xi, scale3*rho12p1 + En[i12+1], color='b', label='1 above |21⟩')

fig.subplots_adjust(hspace=0.05, left=0.13, right=0.95, top=0.94, bottom=0.12)

plt.savefig("density.png")

