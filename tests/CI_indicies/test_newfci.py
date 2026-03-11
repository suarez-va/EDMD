from model_systems.models import GICD
from time_independent.fcidvr import validate_nele_spin, fci_mapping, fci_operator, fci_wfn, FCIDVR
import numpy as np
import time

import itertools


#validate_nele_spin(nele = 2, spin = 'singlet', Sz = 0.0)
#validate_nele_spin(nele = 3, spin = 'doublet', Sz = 0.5)
#validate_nele_spin(nele = 3, spin = 'doublet')
#validate_nele_spin(nele = 2, spin = 'doublet')
#
#exit()

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


R_sub = 8.0
nbo_sub = 25

model = GICD(params)
fcidvr = FCIDVR(model = model, xa = -25.0, xb = 24.9, xN = 50, xbounds = "(-inf,inf)")

ndvr = fcidvr.nxdvr
nfci, map_norm, map_idx = fci_mapping(ndvr = ndvr, nele = 1, spin = 'doublet')
nfci_singlet, map_norm_singlet, map_idx_singlet = fci_mapping(ndvr = ndvr, nele = 2, spin = 'singlet')
nfci_triplet, map_norm_triplet, map_idx_triplet = fci_mapping(ndvr = ndvr, nele = 2, spin = 'triplet')
nfci_doublet, map_norm_doublet, map_idx_doublet = fci_mapping(ndvr = ndvr, nele = 3, spin = 'doublet')

print(nfci)
print(nfci_singlet)
print(nfci_triplet)
print(nfci_doublet)

fcidvr.kernel(R = R_sub, nele = 1, spin = 'doublet', nbo = nbo_sub, derivative_order = 1)
fcidvr.kernel(R = R_sub, nele = 2, spin = 'singlet', nbo = nbo_sub, derivative_order = 1)
fcidvr.kernel(R = R_sub, nele = 2, spin = 'triplet', nbo = nbo_sub, derivative_order = 1)
fcidvr.kernel(R = R_sub, nele = 3, spin = 'doublet', nbo = nbo_sub, derivative_order = 1)

data = np.load('fcidvr_1ele_doublet.npz')
data_singlet = np.load('fcidvr_2ele_singlet.npz')
data_triplet = np.load('fcidvr_2ele_triplet.npz')
data_doublet = np.load('fcidvr_3ele_doublet.npz')

hij = fcidvr.hij(R = R_sub)
gik = fcidvr.gik()
ep, cip = np.linalg.eigh(hij)

Dmap = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if i < k]))
E = ep[map_idx] + fcidvr.model.VR(R = 8.0); E = E[np.argsort(E)]
E_singlet = ep[map_idx_singlet[0]] + ep[map_idx_singlet[1]] + fcidvr.model.VR(R = 8.0); E_singlet = E_singlet[np.argsort(E_singlet)]
E_triplet = ep[map_idx_triplet[0]] + ep[map_idx_triplet[1]] + fcidvr.model.VR(R = 8.0); E_triplet = E_triplet[np.argsort(E_triplet)]
E_doublet = ep[Dmap[0]] + ep[Dmap[1]] + ep[Dmap[2]] + fcidvr.model.VR(R = 8.0); E_doublet = E_doublet[np.argsort(E_doublet)]

print('1ele:')
print(data['En'])
print(E[:nbo_sub])

print('singlet:')
print(data_singlet['En'])
print(E_singlet[:nbo_sub])

print('triplet:')
print(data_triplet['En'])
print(E_triplet[:nbo_sub])

print('doublet:')
print(data_doublet['En'])
print(E_doublet[:nbo_sub])


wfn_0 = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = 0.5, C = data['Cn'][:,0])
wfn_1 = fci_wfn(ndvr = ndvr, nele = 1, spin = 'doublet', Sz = 0.5, C = data['Cn'][:,1])
wfn_S0 = fci_wfn(ndvr = ndvr, nele = 2, spin = 'singlet', Sz = 0.0, C = data_singlet['Cn'][:,0])
wfn_S1 = fci_wfn(ndvr = ndvr, nele = 2, spin = 'singlet', Sz = 0.0, C = data_singlet['Cn'][:,1])
wfn_T0 = fci_wfn(ndvr = ndvr, nele = 2, spin = 'triplet', Sz = 0.0, C = data_triplet['Cn'][:,0])
wfn_T1 = fci_wfn(ndvr = ndvr, nele = 2, spin = 'triplet', Sz = 0.0, C = data_triplet['Cn'][:,1])

#print(np.sum(wfn_0[:,0].conj()*wfn_0[:,0]))
#print(wfn_0[:,0])
#print(wfn_0[:,1])

d00 = np.sum(wfn_0.conj()*wfn_0)
d01 = np.sum(wfn_0.conj()*wfn_1)
d10 = np.sum(wfn_1.conj()*wfn_0)
d11 = np.sum(wfn_1.conj()*wfn_1)
d=np.array([[d00, d01], [d10, d11]])
S00 = np.sum(wfn_S0.conj()*wfn_S0)
S01 = np.sum(wfn_S0.conj()*wfn_S1)
S10 = np.sum(wfn_S1.conj()*wfn_S0)
S11 = np.sum(wfn_S1.conj()*wfn_S1)
S=np.array([[S00, S01], [S10, S11]])
T00 = np.sum(wfn_T0.conj()*wfn_T0)
T01 = np.sum(wfn_T0.conj()*wfn_T1)
T10 = np.sum(wfn_T1.conj()*wfn_T0)
T11 = np.sum(wfn_T1.conj()*wfn_T1)
T=np.array([[T00, T01], [T10, T11]])
print(d)
print(S)
print(T)

exit()

H_singlet = fcidvr.CIoperator(nele = 2, S = 0.0, Sz = 0.0, hij = hij, gik = gik)
H_triplet = fcidvr.CIoperator(nele = 2, S = 1.0, Sz = 0.0, hij = hij, gik = gik)
H_doublet = fcidvr.CIoperator(nele = 3, S = 0.5, Sz = 0.5, hij = hij, gik = gik)

Hmat_singlet = H_singlet.matmat(np.eye(nfci_singlet))
Hsinglet = np.array([[2*hij[0,0]+gik[0,0], np.sqrt(2)*hij[0,1], 0], [np.sqrt(2)*hij[1,0], hij[0,0]+hij[1,1]+gik[0,1], np.sqrt(2)*hij[0,1]], [0, np.sqrt(2)*hij[1,0], 2*hij[1,1]+gik[1,1]]])
print(Hsinglet)
print(Hmat_singlet[:3,:3])


exit()

Dmap = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if i < k]))
#print('doublet map:')
#for i in range(Dmap[0].shape[0]):
#    print(Dmap[0][i], Dmap[1][i], Dmap[2][i])

dat2 = np.load("fcidvr_2ele_singlet.npz")
dat3 = np.load("fcidvr_3ele_doublet.npz")
E2 = dat2['En']
E3 = dat3['En']

Etest3 = ep[Dmap[0]] + ep[Dmap[1]] + ep[Dmap[2]] + fcidvr.model.VR(R = 8.0)
idx3 = np.argsort(Etest3)
Etest3 = Etest3[idx3]

print(E2[:12])
print("no way...")
print(Etest3[:12])
print(E3[:12])



exit()

nfci2, map_norm2, map_idx2 = fcidvr.map_idxping(nele = 2, spin = 'singlet')
nfci3, map_norm3, map_idx3 = fcidvr.map_idxping(nele = 3, spin = 'doublet')
#H2 = fcidvr.CIoperator(nele = 2, spin = 'singlet', hij = hij, gik = gik)
H03 = fcidvr.CIoperator(nele = 3, spin = 'doublet', hij = hij, gik = 0*gik)
H3 = fcidvr.CIoperator(nele = 3, spin = 'doublet', hij = hij, gik = gik)
#Hmat2 = H2.matmat(np.eye(nfci2))
Hmat03 = H03.matmat(np.eye(nfci3))
Hmat3 = H3.matmat(np.eye(nfci3))
#Htest2 = np.array([[2*hij[0,0] + hij[1,1] + 2*gik[0,1] + gik[0,0], hij[0,1]], [hij[1,0], hij[0,0] + 2*hij[1,1] + 2*gik[0,1] + gik[1,1]]])
Htest2 = 2*gik[0,1] + gik[1,1]
Htest5 = 2*gik[0,2] + gik[2,2]
Htest3 = gik[0,1] + gik[0,2] + gik[1,2]
print(np.diag(Hmat3) - np.diag(Hmat03))
print(Htest2)
print(Htest3)
print(Htest5)

#print(Hmat2)
print(Hmat3-Hmat03)
#print(Htest2)
#print(Hmat3[:int(nfci3/2),int(nfci3/2):])
#print(Hmat3[int(nfci3/2):,int(nfci3/2):])
#print(Hmat3[1,5])
exit()


Dmap = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if i < k]))
#print('doublet map:')
#for i in range(Dmap[0].shape[0]):
#    print(Dmap[0][i], Dmap[1][i], Dmap[2][i])

dat3 = np.load("fcidvr_3ele_doublet.npz")
E3 = dat3['En']

Etest3 = ep[Dmap[0]] + ep[Dmap[1]] + ep[Dmap[2]] + fcidvr.model.VR(R = 8.0)
idx3 = np.argsort(Etest3)
Etest3 = Etest3[idx3]

print("no way...")
print(Etest3[:25])
print(E3[:25])

exit()

nfci3, map_norm3, map_idx3 = fcidvr.map_idxping(nele = 3, spin = 'doublet')
H3 = fcidvr.CIoperator(nele = 3, spin = 'doublet', hij = hij)
a = np.random.random((nfci3))
print(a)
print(a.shape)

exit()





nfci2, map_norm2, map_idx2 = fcidvr.map_idxping(nele = 2, spin = 'singlet')
nfci3, map_norm3, map_idx3 = fcidvr.map_idxping(nele = 3, spin = 'doublet')

print('doublet1 map:')
for i in range(map_idx3[0][0].shape[0]):
    print(map_idx3[0][0][i], map_idx3[0][1][i], map_idx3[0][2][i])

print('doublet2 map:')
for i in range(map_idx3[1][0].shape[0]):
    print(map_idx3[1][0][i], map_idx3[1][1][i], map_idx3[1][2][i])

I2 = fcidvr.CIoperator(nele = 2, spin = 'singlet', hij = np.eye(ndvr))
H2 = fcidvr.CIoperator(nele = 2, spin = 'singlet', hij = hij)
I3 = fcidvr.CIoperator(nele = 3, spin = 'doublet', hij = np.eye(ndvr))
H3 = fcidvr.CIoperator(nele = 3, spin = 'doublet', hij = hij)
Imat2 = I2.matmat(np.eye(nfci2))
Hmat2 = H2.matmat(np.eye(nfci2))
#Imat3 = I3.matmat(np.eye(nfci3))
Hmat3 = H3.matmat(np.eye(nfci3))
#print(H3.matvec(np.array([1,0])))

#print(np.max(np.abs(Imat2 - np.diag(np.diag(Imat2)))))
#print(np.diag(Imat2))
#print(np.max(np.abs(Imat3 - np.diag(np.diag(Imat3)))))
#print(np.diag(Imat3))

print(hij)
print(np.max(np.abs(hij - hij.conj().T)))
print(Hmat2)
print(np.max(np.abs(Hmat2 - Hmat2.conj().T)))
print(Hmat3)
print(np.max(np.abs(Hmat3 - Hmat3.conj().T)))
#error = Hmat3 - Hmat3.conj().T
#ierror = np.argwhere(error != 0)
#print(ierror)

#print(Hmat3[int(nfci3/2):,:int(nfci3/2)])
#print(Hmat3[:int(nfci3/2),int(nfci3/2):])

#Htest3 = np.array([[2*hij[0,0] + hij[1,1], hij[0,1]], [hij[1,0], hij[0,0] + 2*hij[1,1]]])
#print(Htest3)
#exit()
#etest3, c3 = np.linalg.eigh(Htest3)


ep, cip = np.linalg.eigh(hij)
E2, C2 = np.linalg.eigh(Hmat2)
idx2 = np.argsort(E2)
E2 = E2[idx2]
E3, C3 = np.linalg.eigh(Hmat3)
#E3, C3 = np.linalg.eigh(Hmat3[:int(nfci3/2),:int(nfci3/2)])
idx3 = np.argsort(E3)
E3 = E3[idx3]

#print(ep)
#print(E2)
#print(E3)
#print(2*ep[0] + ep[1])
#print(ep[0] + 2*ep[1])
#exit()
#print(etest3)
#exit()

#ep, cip = np.linalg.eigh(hij)
#E2, C2 = np.linalg.eigh(Hmat2)
#idx2 = np.argsort(E2)
#E2 = E2[idx2]
##E3, C3 = np.linalg.eigh(Hmat3)
#E3, C3 = np.linalg.eigh(Hmat3)
#idx3 = np.argsort(E3)
#E3 = E3[idx3]

Dmap = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if i < k]))
print('doublet map:')
for i in range(Dmap[0].shape[0]):
    print(Dmap[0][i], Dmap[1][i], Dmap[2][i])


Etest2 = ep[map_idx2[0]] + ep[map_idx2[1]]
idx2 = np.argsort(Etest2)
Etest2 = Etest2[idx2]
Etest3 = ep[Dmap[0]] + ep[Dmap[1]] + ep[Dmap[2]]
idx3 = np.argsort(Etest3)
Etest3 = Etest3[idx3]

print("first 2e:")
print(Etest2)
print(E2)

print("no way...")
print(Etest3)
print(E3)

exit()
print(ep[0])
print(ep[1])
print(2*ep[0]+ep[1])

exit()


#print(Hmat)
#print(np.max(np.abs(Hmat - Hmat.conj().T)))
#error = Hmat - Hmat.conj().T
#ierror = np.argwhere(error != 0)
#print(ierror)

#exit()

Na = int((ndvr) * (ndvr - 1) / 2)
Nb = ndvr
Nfci = Na * Nb
Nquartet = int((ndvr) * (ndvr - 1) * (ndvr - 2) / 6)
#Ndoublet = int((ndvr) * (ndvr - 1) * (ndvr - 2) / 6 + (ndvr) * (ndvr - 1) / 2)
Ndoublet = int((ndvr + 1) * (ndvr) * (ndvr - 1) / 6)
print(Nfci)
#print(Nquartet)
#print(nfci3)
#print(Ndoublet)
print(Nquartet + 2 * Ndoublet)
exit()

dat1 = np.load("fcidvr_1ele_doublet.npz")
dat2 = np.load("fcidvr_2ele_singlet.npz")
dat3 = np.load("fcidvr_3ele_doublet2.npz")

#ep = dat1['En']
hij = fcidvr.hij(R = 8.0)
ep, cip = np.linalg.eigh(hij)

H = fcidvr.CIoperator(nele = 3, spin = 'doublet2', hij = hij)
nfci3 = H.shape[0]
I = fcidvr.CIoperator(nele = 3, spin = 'doublet2', hij = np.eye(ndvr))
Hmat = H.matmat(np.eye(nfci3))
Imat = I.matmat(np.eye(nfci3))
#print(np.diag(Imat))
#exit()

Hmat = H.matmat(np.eye(nfci3))
#print(Hmat - Hmat.conj().T)
hij = fcidvr.hij(R = 8.0)
ep, cip = np.linalg.eigh(hij)
E, C = np.linalg.eigh(Hmat)

En2 = dat2['En']
En3 = dat3['En']
nfci2, map_norm2, map_idx2 = fcidvr.map_idxping(nele = 2, spin = 'singlet')
nfci3, map_norm3, map_idx3 = fcidvr.map_idxping(nele = 3, spin = 'doublet2')
Etest2 = ep[map_idx2[0]] + ep[map_idx2[1]] + fcidvr.model.VR(R = 8.0)
idx2 = np.argsort(Etest2)
Etest2 = Etest2[idx2]
Etest3 = ep[map_idx3[0]] + ep[map_idx3[1]] + ep[map_idx3[2]] + fcidvr.model.VR(R = 8.0)
idx3 = np.argsort(Etest3)
Etest3 = Etest3[idx3]



#print(Etest2)
#print(En2)

print("Now 3:")

print(Etest3)
#print(En3)
print(E + fcidvr.model.VR(R = 8.0))

exit()
print(f"VR: {fcidvr.model.VR(R = 8.0)}")

print("Now Look at Combs:")

for i in range(nfci3):
    #print(map_idx3[0][i], map_idx3[1][i], map_idx3[2][i])
    print(f"{(int(map_idx3[0][i]), int(map_idx3[1][i]), int(map_idx3[2][i]))}: {map_norm3[map_idx3[0][i], map_idx3[1][i], map_idx3[2][i]]**2}")

#print(map_norm3)
#temp = map_norm3.swapaxes(0,1) + map_norm3.transpose(2,0,1)
#print(map_norm3)

exit()

#nfci, map_norm, map_idx = fcidvr.map_idxping(nele = 2, spin = 'singlet')
#fcidvr.kernel(R = 8.0, nele = 1, spin = 'doublet', nbo = 40)
#fcidvr.kernel(R = 8.0, nele = 2, spin = 'singlet', nbo = 40)
#fcidvr.kernel(R = 8.0, nele = 2, spin = 'triplet', nbo = 40)

ndvr = fcidvr.nxdvr
#nfci, map_norm, map_idx = fcidvr.map_idxping(nele = 2, spin = 'singlet')
#operator = fcidvr.CIoperator(nele = 2, spin = 'singlet', hij = np.eye(ndvr))
nfci, map_norm, map_idx = fcidvr.map_idxping(nele = 3, spin = 'doublet2')
operator = fcidvr.CIoperator(nele = 3, spin = 'doublet2', hij = np.eye(ndvr))
Cmat = np.eye(nfci)
Cres = operator.matmat(Cmat)
print(np.diag(Cres))


exit()
nfci, map_norm, map_idx = fcidvr.map_idxping(nele = 3, spin = 'doublet2')
print(nfci)
ndvr = fcidvr.nxdvr
print((ndvr) * (ndvr - 1) * (ndvr - 2) / 6)
print((ndvr + 1) * (ndvr) * (ndvr - 1) / 6)
print((ndvr + 2) * (ndvr) * (ndvr + 1) / 6)
#print(map_norm)
print(map_idx.shape)
#print(map_idx[0].nbytes / 1024**3)
#print(map_idx[0])

#fcidvr.kernel(R = 8.0, nele = 1, spin = 'singlet', nbo = 25)
#fcidvr.kernel(R = 8.0, nele = 2, spin = 'doublet', nbo = 25)
#fcidvr.kernel(R = 8.0, nele = 3, spin = 'doublet', nbo = 25)

