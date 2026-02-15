import numpy as np

import itertools

a = np.array([0,0,0,0,1,1,1,1])
b = np.array([0,0,1,1,0,0,1,1])
c = np.array([0,1,0,1,0,1,0,1])
d = np.random.random((2,2))
e = d[a,b]
print(d)
print(e)
exit()

npts = 5
a = np.arange(npts**3)
b = a.reshape(npts, npts, npts)
c = b.transpose(2,0,1)
d = b.swapaxes(1,2).swapaxes(0,1)
i = 2
j = 3
k = 1
print(b[j,k,i])
print(c[i,j,k])
print(d[i,j,k])

e = b.transpose(1,2,0)
print(b[k,i,j])
print(e[i,j,k])

exit()

ndvr = 5


CImap1 = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if (i < j < k) or (i < j and j == k)]))
CImap2 = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if (i < j < k) or (i == j and j < k)]))

Nfci = int((ndvr)**2 * (ndvr - 1) / 2)
Nquartet = int((ndvr) * (ndvr - 1) * (ndvr - 2) / 6)
Ndoublet1 = int((ndvr + 1) * (ndvr) * (ndvr - 1) / 6)
Ndoublet2 = int((ndvr + 1) * (ndvr) * (ndvr - 1) / 6)
Ndoublet = Ndoublet1 + Ndoublet2
print(Nfci, (Nquartet + Ndoublet))

print('doublet1 map:')
for i in range(Ndoublet1):
    print(CImap1[0][i], CImap1[1][i], CImap1[2][i])

print('doublet2 map:')
for i in range(Ndoublet2):
    print(CImap2[0][i], CImap2[1][i], CImap2[2][i])

d = np.eye(Ndoublet1)

print('S11:')
Smat11 = np.zeros((Ndoublet1, Ndoublet1), dtype=np.float64)
for J in range(Ndoublet1):
    for I in range(Ndoublet1):
        i = CImap1[0][I]
        k = CImap1[1][I]
        m = CImap1[2][I]
        j = CImap1[0][J]
        l = CImap1[1][J]
        n = CImap1[2][J]
        Smat11[I,J] += d[i,j]*d[k,l]*d[m,n] + d[i,j]*d[k,n]*d[m,l] - 0.5*d[i,l]*d[k,j]*d[m,n] - 0.5*d[i,l]*d[k,n]*d[m,j] - 0.5*d[i,n]*d[k,j]*d[m,l] - 0.5*d[i,n]*d[k,l]*d[m,j]
print(np.max(np.abs(Smat11 - np.diag(np.diag(Smat11)))))
print(np.diag(Smat11))

print('S22:')
Smat22 = np.zeros((Ndoublet2, Ndoublet2), dtype=np.float64)
for J in range(Ndoublet2):
    for I in range(Ndoublet2):
        i = CImap2[0][I]
        k = CImap2[1][I]
        m = CImap2[2][I]
        j = CImap2[0][J]
        l = CImap2[1][J]
        n = CImap2[2][J]
        #Smat22[I,J] += d[i,j]*d[k,l]*d[m,n] - d[i,j]*d[k,n]*d[m,l] + 0.5*d[i,l]*d[k,j]*d[m,n] - 0.5*d[i,l]*d[k,n]*d[m,j] - 0.5*d[i,n]*d[k,j]*d[m,l] + 0.5*d[i,n]*d[k,l]*d[m,j]
        Smat22[I,J] += d[i,j]*d[k,l]*d[m,n] - d[i,j]*d[k,n]*d[m,l] - (0.5*d[i,l]*d[k,j]*d[m,n] - 0.5*d[i,l]*d[k,n]*d[m,j] - 0.5*d[i,n]*d[k,j]*d[m,l] + 0.5*d[i,n]*d[k,l]*d[m,j])
print(np.max(np.abs(Smat22 - np.diag(np.diag(Smat22)))))
print(np.diag(Smat22))

print('S12:')
Smat12 = np.zeros((Ndoublet1, Ndoublet2), dtype=np.float64)
for J in range(Ndoublet2):
    for I in range(Ndoublet1):
        i = CImap1[0][I]
        k = CImap1[1][I]
        m = CImap1[2][I]
        j = CImap2[0][J]
        l = CImap2[1][J]
        n = CImap2[2][J]
        Smat12[I,J] += 0.5*np.sqrt(3)*d[i,l]*d[k,j]*d[m,n] + 0.5*np.sqrt(3)*d[i,l]*d[k,n]*d[m,j] - 0.5*np.sqrt(3)*d[i,n]*d[k,j]*d[m,l] - 0.5*np.sqrt(3)*d[i,n]*d[k,l]*d[m,j]
#print(np.max(np.abs(Smat22)))
#print(np.diag(Smat22))
print(Smat12)

a = np.arange(7)
print(a)
b = a
b *= 8
print(a)
print(b)

print(np.diag(d)[:10].shape)
print(np.diag(d)[10:].shape)

exit()

def triu_indices_3d(ndvr):
    triplets = list(itertools.combinations_with_replacement(range(ndvr), 3))

    i, j, k = zip(*triplets)

    return np.array(i), np.array(j), np.array(k)

def triu_3d_no_diagonal(ndvr):
    triplets = [
        (i, j, k)
        for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3)
        if k > i
    ]

    i, j, k = zip(*triplets)
    return np.array(i), np.array(j), np.array(k)

def triu_strict_3d(ndvr):
    i_list = []
    j_list = []
    k_list = []

    for i, j in itertools.combinations_with_replacement(range(ndvr), 2):
        for k in range(j + 1, ndvr):
            i_list.append(i)
            j_list.append(j)
            k_list.append(k)

    return np.array(i_list), np.array(j_list), np.array(k_list)

def Sikmjln(ndvr):
    delta = np.eye(ndvr, dtype=float)  # Kronecker delta δ_ij

    delta_ij = delta[:, None, None, :, None, None]
    delta_kl = delta[None, :, None, None, :, None]
    delta_mn = delta[None, None, :, None, None, :]

    delta_il = delta[:, None, None, None, :, None]
    delta_kj = delta[None, :, None, :, None, None]

    delta_kn = delta[None, :, None, None, None, :]
    delta_ml = delta[None, None, :, None, :, None]

    delta_mj = delta[None, None, :, :, None, None]
    delta_in = delta[:, None, None, None, None, :]

    S = (
        delta_ij * delta_kl * delta_mn
        + delta_il * delta_kj * delta_mn
        - 0.5 * delta_ij * delta_kn * delta_ml
        - 0.5 * delta_il * delta_kn * delta_mj
        - 0.5 * delta_in * delta_kj * delta_ml
        - 0.5 * delta_in * delta_kl * delta_mj
    )

    return S

ndvr = 4
print(ndvr**6)

S = Sikmjln(ndvr)

print(S[1,3,2,1,2,3])
exit()
print(S[0,0,0,0,0,0]) #0
print(S[0,0,1,0,0,1]) #2
print(S[0,1,1,0,1,1]) #1/2

#CImap = triu_indices_3d(ndvr)
CImap = triu_3d_no_diagonal(ndvr)
#CImap = triu_strict_3d(ndvr)
nfci = CImap[0].shape[0]
#CImap = (np.append(CImap[0],0), np.append(CImap[1],1), np.append(CImap[2],0))
#nfci += 1

for i in range(nfci):
    print(CImap[0][i], CImap[1][i], CImap[2][i])

map_ij = CImap[0]
map_kl = CImap[1]
map_mn = CImap[2]
nfci = map_ij.shape[0]

Snm = np.zeros((nfci, nfci), dtype=np.float64)
Snm = S[map_ij[:,None], map_kl[:,None], map_mn[:,None], map_ij[None,:], map_kl[None,:], map_mn[None,:]]
print(Snm)
exit()
print(Snm.shape)
print(np.max(np.abs(Snm - np.diag(np.diag(Snm)))))
i0 = np.argwhere(np.diag(Snm) == 0)[:,0]

#for i in i0:
#    print(CImap[0][i], CImap[1][i], CImap[2][i])

for n in range(nfci):
    print(f"({int(map_ij[n])}, {int(map_kl[n])}, {int(map_mn[n])}): {np.diag(Snm)[n]}")
    #if np.diag(Snm)[n] != 0.0:
    #    print(f"({int(map_ij[n])}, {int(map_kl[n])}, {int(map_mn[n])})")

i = 1
k = 3
m = 1
print((i,k,m))
for n in range(nfci):
    print(f"({int(map_ij[n])}, {int(map_kl[n])}, {int(map_mn[n])}): {S[i,k,m,map_ij[n],map_kl[n],map_mn[n]]}")


exit()

def two_elec_basis(nxdvr):
    nij = (1 - (1 - 1 / np.sqrt(2)) * np.eye(nxdvr))
    #map_ij, map_kl = np.triu_indices(nxdvr, k=0)
    Cdeltamap = np.triu_indices(nxdvr, k=0)
    return nij, CImap
ndvr = 5
n, map = two_elec_basis(ndvr)
nfci = map[0].shape[0]

print(nfci)

C = np.exp(-1j * np.random.uniform(0, 2*np.pi, size=nfci))
Cij = np.zeros((ndvr, ndvr), dtype=np.complex128)
#Cij[map[0],map[1]] = C
Cij[map] = C
print(Cij)
