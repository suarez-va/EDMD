import numpy as np
from scipy.linalg import expm
from models.model_utils import matmat
from models.two_electron_diatomic import TEGD

params = {
    "aR": 0.0,
    "bR": 0.0001,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.0,
    "bee": 0.0001,
}

model = TEGD(a=-200, b=200, N=750, bounds="(-inf,inf)", spin="singlet", model_params=params)
#model = TEGD(a=-200, b=200, N=500, bounds="(-inf,inf)", spin="singlet", model_params=params)
ep, cip = model.solve_mos(R = 8.0)
ci1 = cip[:,1]
ci2 = cip[:,2]
S12 = 1.0 / np.sqrt(2) * (ci1[:,None] * ci2[None,:] + ci2[:,None] * ci1[None,:])

St0 = model.wfn_to_vec(S12)
Hop = model.H(R = 8.0)

Cn = St0
print(np.sum(Cn.conj() * Cn))
dt = 0.01
nrk4 = 50000
nprint = 1000
xi = model.xi()
tpts = int(np.ceil(nrk4 / nprint))
time = dt * nprint * np.arange(tpts)
rhot = np.zeros((model.ndvr, tpts), dtype=np.float64)
t = 0
for i in range(nrk4):
    if i % nprint == 0:
        print(time[t])
        print(np.sum(Cn.conj() * Cn))
        Cij = model.vec_to_wfn(Cn)
        rhot[:,t] = 2*np.sum(Cij.conj() * Cij, axis=1).real
        t += 1
    k1 = -1j * Hop.matvec(Cn)
    k2 = -1j * Hop.matvec(Cn + 0.5 * dt * k1)
    k3 = -1j * Hop.matvec(Cn + 0.5 * dt * k2)
    k4 = -1j * Hop.matvec(Cn + dt * k3)
    Cn += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
np.savez('timedata', t=time, xi=xi, rhot=rhot)
exit()


HS = Hop.matvec(St0)

print(St0.shape)
print(HS.shape)

print(np.sum(St0.conj() * St0))
print(np.sum(HS.conj() * HS))

modelS = TEGD(a=-60, b=60, N=125, bounds="(-inf,inf)", spin="singlet", model_params=params)
modelT = TEGD(a=-60, b=60, N=125, bounds="(-inf,inf)", spin="triplet", model_params=params)

ep, cip = modelS.solve_mos(R = 8.0)

mospec = np.load('mospec.npz') 

cip = mospec["cip"]
ci1 = cip[:,1]
ci2 = cip[:,2]
S11 = 1.0 / 2.0 * (ci1[:,None] * ci1[None,:] + ci1[:,None] * ci1[None,:])
S12 = 1.0 / np.sqrt(2) * (ci1[:,None] * ci2[None,:] + ci2[:,None] * ci1[None,:])
T12 = 1.0 / np.sqrt(2) * (ci1[:,None] * ci2[None,:] - ci2[:,None] * ci1[None,:])

#print(np.sum(np.sum(S11.conj() * S11, axis=1)))
#print(np.sum(np.sum(S12.conj() * S12, axis=1)))

Sijn = np.zeros((modelS.ndvr, modelS.ndvr, 2), dtype=np.complex128)
Sijn[:,:,0] = S11; Sijn[:,:,1] = S12

#print(Sijn.shape)
SIn = modelS.wfn_to_vec(Sijn)
#print(Sijn.shape)
#print(SIn.shape)

#print(np.sum(SIn[:,0].conj() * SIn[:,0]))
#print(np.sum(SIn[:,1].conj() * SIn[:,1]))

TI = modelT.wfn_to_vec(T12)
#TI = modelT.wfn_to_vec(T12[:,:,None])
print(TI.shape)
print(np.sum(TI.conj() * TI))
print(TI.shape)


Sijn2 = modelS.vec_to_wfn(SIn)
print(np.max(np.absolute(Sijn - Sijn2)))

Tij2 = modelT.vec_to_wfn(TI)
print(np.max(np.absolute(Tij2 - T12)))

exit()

CIn = modelS.solve_wfn(R = 8.0, nbo = 75)
Wab50n2 = np.matmul(CIn.conj().T, matmat(modelS.W(acap=-50.0, bcap=50.0, ncap=2), CIn))
np.savez("boops", Wab50n2=Wab50n2)

#eta = 3e-04
#eta = 1e-04
eta =1e-10

#mospec = np.load('mospec.npz') 
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


