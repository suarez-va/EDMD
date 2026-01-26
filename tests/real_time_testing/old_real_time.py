import numpy as np
from scipy.linalg import expm

#eta = 3e-04
#eta = 1e-04
eta =1e-10

mospec = np.load('mospec.npz') 
eigsolve = np.load('eigspec.npz')
boops = np.load('boops.npz')
Hnm = np.diag(eigsolve['En'])
Wnm = boops["Wab50n2"]

Heta = Hnm - 1j * eta * Wnm

xi = eigsolve["xi"]
cip = mospec["cip"]
ci1 = cip[:,1]
ci2 = cip[:,2]
Cijt0 = 1.0 / np.sqrt(2) * (ci1[:,None] * ci2[None,:] + ci2[:,None] * ci1[None,:])

Cijn = eigsolve["Cijn"]
ndvr = Cijn.shape[0]
nbo = Cijn.shape[2]

Ct0 = np.zeros((nbo), dtype=np.complex128)
for n in range(nbo):
    #Ct0[n] = np.sum(np.sum(Cijn[:,:,n].conj() * Cijt0, axis=1))
    Ct0[n] = np.matmul(Cijn[:,:,n].reshape(ndvr**2).conj().T, Cijt0.reshape(ndvr**2))

print(Ct0[43])
print(np.argmax(np.abs(Ct0)))
#print(np.matmul(Ct0.conj().T, Ct0))

tpts = 101
t = np.linspace(0,500,tpts)
Ct = np.zeros((nbo, tpts), dtype=np.complex128)
for i, time in enumerate(t):
    print(i)
    Ct[:,i] = np.matmul(expm(-1j*Heta*time), Ct0)


#Cijt = np.zeros((ndvr, ndvr, tpts), dtype=np.complex128)
#for n in range(nbo):
#    for i in range(tpts):
#        Cijt[:,:,i] += Ct[n,i] * Cijn[:,:,n]
Cijt = np.tensordot(Cijn, Ct, axes=([2], [0]))
#Cijt = np.zeros((ndvr, ndvr, tpts), dtype=np.complex128)
#for n in range(nbo):
#    print(n)
#    Cijt += Ct[None,None,n,:] * Cijn[:,:,n,None]

rhot = 2*np.sum(Cijt.conj() * Cijt, axis=1).real

print(np.sum(rhot[:,0]/2))

np.savez('timedata', t=t, xi=xi, rhot=rhot)
exit()

#print(rhot.shape)
#print(Heta)
#print(np.matmul(Ct.conj().T, Ct))
#print(np.sum(rhot))


