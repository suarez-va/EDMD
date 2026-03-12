import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from time_independent.fcidvr import fci_wfn, fci_density

fcidvr_data1 = np.load('Ln/L0/fcidvr_1ele_doublet.npz')
fcidvr_data2 = np.load('Ln/L0/fcidvr_2ele_singlet.npz')
nbo1 = fcidvr_data1['En'].shape[0]
nbo2 = fcidvr_data2['En'].shape[0]

imax = 20
Ln = np.loadtxt("Ln/Ln.dat", dtype=np.float64)[:imax]
Lpts = Ln.shape[0]

epn = np.zeros((nbo1, Lpts), dtype=np.float64)
Enn = np.zeros((nbo2, Lpts), dtype=np.float64)

i00 = np.zeros((Lpts), dtype=int)
i01 = np.zeros((Lpts), dtype=int)
i11 = np.zeros((Lpts), dtype=int)
i12 = np.zeros((Lpts), dtype=int)
i22 = np.zeros((Lpts), dtype=int)

for n in range(Lpts):
    print(n)
    sub_dir_curr = f"Ln/L{n}/"
    fcidvr_data1 = np.load(sub_dir_curr + 'fcidvr_1ele_doublet.npz')
    fcidvr_data2 = np.load(sub_dir_curr + 'fcidvr_2ele_singlet.npz')

    epn[:,n] = fcidvr_data1['En']
    Enn[:,n] = fcidvr_data2['En']

    xi = fcidvr_data1["xi"]
    ndvr = xi.shape[0]
    cp = fcidvr_data1['Cn']
    Cn = fcidvr_data2['Cn']

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
    for N in range(nbo2):
        wfn = fci_wfn(ndvr = ndvr, nele = 2, spin = 'singlet', Sz = 0, C = Cn[:,N])
        C00[N] = np.sum(wfn.conj()*S00)
        C01[N] = np.sum(wfn.conj()*S01)
        C11[N] = np.sum(wfn.conj()*S11)
        C12[N] = np.sum(wfn.conj()*S12)
        C22[N] = np.sum(wfn.conj()*S22)

    i00[n] = np.argmax(np.abs(C00))
    i01[n] = np.argmax(np.abs(C01))
    i11[n] = np.argmax(np.abs(C11))
    i12[n] = np.argmax(np.abs(C12))
    i22[n] = np.argmax(np.abs(C22))
    #print(f'S00: i={i00[n]}, |C|={np.abs(C00[i00[n]])}')
    #print(f'S01: i={i01[n]}, |C|={np.abs(C01[i01[n]])}')
    #print(f'S11: i={i11[n]}, |C|={np.abs(C11[i11[n]])}')
    print(f'S12: i={i12[n]}, |C|={np.abs(C12[i12[n]])}')
    #print(f'S22: i={i22[n]}, |C|={np.abs(C22[i22[n]])}')

epn = np.zeros((nbo1, Lpts), dtype=np.float64)
Enn = np.zeros((nbo2, Lpts), dtype=np.float64)

i00 = np.zeros((Lpts), dtype=int)
i01 = np.zeros((Lpts), dtype=int)
i11 = np.zeros((Lpts), dtype=int)
i12 = np.zeros((Lpts), dtype=int)
i22 = np.zeros((Lpts), dtype=int)

np.savez(f'Ldata', Ln=Ln, epn=epn, Enn=Enn, i00=i00, i01=i01, i11=i11, i12=i12, i22=i22)

