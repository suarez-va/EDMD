import numpy as np
from models.two_electron_screened_diatomic import TESD, generate_Hele, generate_dHele, generate_dipole, generate_cap, Vnuc


#xpts = 4
#
#M = np.random.random((xpts,xpts))
#
#map = np.full((xpts, xpts), -1)
#m = 0
#for i in range(xpts):
#    for j in range(i+1,xpts):
#        map[i,j] = m
#        m += 1
#
#nstates = int((xpts - 1) * xpts / 2)
#ovlp = np.zeros((nstates,nstates))
#H = np.zeros((nstates,nstates))
#for i in range(xpts):
##    for k in range(i+1,xpts):
#        for j in range(xpts):
#            for l in range(j+1,xpts):
#                dij = 1 if i == j else 0
#                dkl = 1 if k == l else 0
#                dil = 1 if i == l else 0
#                dkj = 1 if k == j else 0
#                ovlp[map[i,k],map[j,l]] = dij*dkl-dil*dkj
#                H[map[i,k],map[j,l]] = M[i,j]*dkl+dij*M[k,l]-M[i,l]*dkj-dil*M[k,j]
#
#print(ovlp)
#print(H)
#
#H2 = np.zeros((nstates,nstates))
#for i in range(xpts):
#    for k in range(i+1,xpts):
#        dik = 1 if i == k else 0
#        H2[map[i,k],map[i,k]] = M[i,i] + M[k,k] - M[i,k]*dik - M[k,i]*dik
#for i in range(xpts):
#    for k in range(i+1,xpts):
#        for l in range(i+1,xpts):
#            if k == l:
#                pass
#            else:
#                H2[map[i,k],map[i,l]] = M[k,l]
#for i in range(xpts):
#    for k in range(i+1,xpts):
#        for j in range(xpts):
#            if i == j:
#                pass
#            else:
#                H2[map[i,k],map[i,l]] = M[i,j]
#
#print(H2)
#
#
#
#exit()
xpts = 4

map = np.full((xpts, xpts), -1)
m = 0
for i in range(xpts):
    for j in range(i,xpts):
        map[i,j] = m
        m += 1

print(map)

nstates = int((xpts + 1) * xpts / 2)
print(nstates)
ovlp = np.zeros((nstates,nstates))
for i in range(xpts):
    for k in range(i,xpts):
        for j in range(xpts):
            for l in range(j,xpts):
                print(map[i,k], map[j,l])
                dij = 1 if i == j else 0
                dkl = 1 if k == l else 0
                dil = 1 if i == l else 0
                dkj = 1 if k == j else 0
                ovlp[map[i,k],map[j,l]] = dij*dkl+dil*dkj
                #ovlp[map[j,l],map[i,k]] = dij*dkl+dil*dkj
                #if i == j and j == k:
                if i == j and k == l and i == k:
                    ovlp[map[i,k],map[j,l]] *= 0.5   
#                if i != j and k != l:
#                    print((i,l))
#                    print((k,j))
print(ovlp)

print(np.diagonal(ovlp))

