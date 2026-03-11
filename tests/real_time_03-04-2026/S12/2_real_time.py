import numpy as np
from time_independent.fcidvr import fci_wfn
from time_dependent.fci_lindblad_cap import FCILindbladCAP

fcidvr_data1 = np.load('fcidvr_1ele_doublet.npz')
fcidvr_data_cap1 = np.load('fcidvr_1ele_doublet_cap.npz')
fcidvr_data_density1 = np.load('fcidvr_1ele_doublet_0.5Sz_density.npz')
fcidvr_data_dyson = np.load('fcidvr_1ele_doublet_0.5Sz_2ele_singlet_0.0Sz_dyson.npz')
fcidvr_data2 = np.load('fcidvr_2ele_singlet.npz')
fcidvr_data_cap2 = np.load('fcidvr_2ele_singlet_cap.npz')
fcidvr_data_density2 = np.load('fcidvr_2ele_singlet_0.0Sz_density.npz')

xi = fcidvr_data1["xi"]
ep = fcidvr_data1["En"]
cp = fcidvr_data1["Cn"]
wi = fcidvr_data_cap1["wi"]
wpq = fcidvr_data_cap1["Wnm"]
Pipq = fcidvr_data_density1["Pisnm"][:,0,:,:]
Dipn = fcidvr_data_dyson["Dispn"][:,1,:,:]
En = fcidvr_data2["En"]
Cn = fcidvr_data2["Cn"]
Wnm = fcidvr_data_cap2["Wnm"]
Pinm = 2*fcidvr_data_density2["Pisnm"][:,0,:,:]

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
#wfn_init = S00
#wfn_init = S01
#wfn_init = 1.0/np.sqrt(2)*(S01 + S12)
wfn_init = S12
N0 = 0j
Ppq = np.zeros((nbo1, nbo1), dtype=np.complex128)
C0 = np.zeros((nbo2), dtype=np.complex128)

for n in range(nbo2):
    wfn = fci_wfn(ndvr = ndvr, nele = 2, spin = 'singlet', Sz = 0, C = Cn[:,n])
    C0[n] = np.sum(wfn.conj()*wfn_init)
print(np.sum(C0.conj()*C0))

eta = 1e-04
fcilindbladcap = FCILindbladCAP(xi, wi, eta, ep, wpq, Pipq, Dipn, En, Wnm, Pinm)
fcilindbladcap.N0 = N0
fcilindbladcap.Ppq = Ppq
fcilindbladcap.Cn = C0
fcilindbladcap.kernel(timestep = 0.1, nsteps = 100000, nprint = 10)

