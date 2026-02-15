import numpy as np
from scipy.linalg import expm
from models.model_utils import matmat, solve_cap
from models.two_electron_diatomic import TEGD
from edmd import EDMD
import time

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

#def __init__(self, eta, wi, Ek, Dikn, En, Wnm, Cn0):
model = TEGD(a=-196.7/2.0, b=196.7/2.0, N=750, bounds="(-inf,inf)", spin="singlet", model_params=params)
mospec = np.load('mospec.npz') 
moops = np.load('moops.npz')
bospec = np.load('bospec.npz')
boops = np.load('boops.npz')

eta = 1e-04
nmo = 25
nbo = 222
ep = mospec['ep'][:nmo]
cip = mospec['cip'][:,:nmo]
wk = moops['wk']
wpq = moops['wpq'][:nmo,:nmo]
En = bospec['En'][:nbo]
Cijn = bospec['Cijn'][:,:,:nbo]
Wnm = boops['Wnm'][:nbo,:nbo]

edmd = EDMD(model, eta, wk, ep, cip, wpq, En, Cijn, Wnm)

ndvr = wk.shape[0]
ci1 = cip[:,1]
ci2 = cip[:,2]
Cij0 = 1.0 / np.sqrt(2) * (ci1[:,None] * ci2[None,:] + ci2[:,None] * ci1[None,:])
Cn0 = np.zeros((nbo), dtype=np.complex128)
for n in range(nbo):
    Cn0[n] = np.matmul(Cijn[:,:,n].reshape(ndvr**2).conj().T, Cij0.reshape(ndvr**2))

edmd.kernel(Cn0, timestep = 0.1, nsteps = 100000, nprint = 10)

exit()
hij = mospec['hij']
wij = moops["wab50n2"]

Hnm = np.diag(eigspec['En'])[:nbo2,:nbo2]
Wnm = boops["Wab50n2"][:nbo2,:nbo2]
#Hnm = np.diag(eigspec['En'])
#Wnm = boops["Wab50n2"]


exit()

ea, xial, xiar = solve_cap(hij, wij, eta)
xia = xiar
xai_inv = xial.conj().T

Ea, Xnal, Xnar = solve_cap(Hnm, Wnm, eta)
Xna = Xnar
Xan_inv = Xnal.conj().T

cip = mospec["cip"]
ci1 = cip[:,1]; ci2 = cip[:,2]
S12 = 1.0 / np.sqrt(2) * (ci1[:,None] * ci2[None,:] + ci2[:,None] * ci1[None,:])
Cijn = eigspec["Cijn"][:,:,:nbo2]
#Cijn = eigspec["Cijn"]
ndvr = Cijn.shape[0]
nbo = Cijn.shape[2]

Ctilde=np.einsum('ai,kin,nb->kab', xai_inv, Cijn, Xna, optimize=True)

Ct0 = np.zeros((nbo), dtype=np.complex128)
for n in range(nbo):
    Ct0[n] = np.matmul(Cijn[:,:,n].reshape(ndvr**2).conj().T, S12.reshape(ndvr**2))
Pt0 = Ct0[:,None] * Ct0[None,:].conj()

Ptilde=np.matmul(Xan_inv, np.matmul(Pt0, Xan_inv.conj().T))

#de=ea[:,None] - ea[None,:].conj()
#dE=Ea[:,None] - Ea[None,:].conj()
#
#print(np.max(np.abs(de)))
#print(np.max(np.abs(dE)))
#exit()

tpts = 3
t = np.linspace(0,500,tpts)
padt = np.zeros((ndvr, ndvr, tpts), dtype=np.complex128)
kidx = np.argwhere(np.abs(wk)>=0.0001)[:,0]
for d in range(ndvr):
    print(f"d = {d}")
    for a in range(ndvr):
        print(f"a = {a}")
        padt[a,d,:] = -4j*eta*(wk[kidx,None,None,None]*Ctilde[kidx,a,:,None,None]*Ptilde[None,:,:,None]*Ctilde.conj()[kidx,None,d,:,None]*np.exp(-1j*(Ea[None,:,None,None]-Ea.conj()[None,None,:,None])*t[None,None,None,:])/((ea[a]-ea.conj()[d]) - (Ea[None,:,None,None]-Ea.conj()[None,None,:,None]))).sum(axis=(0,1,2))
        #padt1 = -4j*eta*(wk[kidx,None,None,None]*Ctilde[kidx,a,:,None,None]*Ptilde[None,:,:,None]*Ctilde.conj()[kidx,None,d,:,None]*np.exp(-1j*(Ea[None,:,None,None]-Ea.conj()[None,None,:,None])*t[None,None,None,:])/((ea[a]-ea.conj()[d]) - (Ea[None,:,None,None]-Ea.conj()[None,None,:,None]))).sum(axis=(0,1,2))
        #padt2 = -4j*eta*(wk[kidx,None,None,None]*Ctilde[kidx,a,:,None,None]*Ptilde[None,:,:,None]*Ctilde.conj()[kidx,None,d,:,None]*np.exp(-1j*(Ea[None,:,None,None]-Ea.conj()[None,None,:,None])*t[None,None,None,:])/(-(Ea[None,:,None,None]-Ea.conj()[None,None,:,None]))).sum(axis=(0,1,2))
        #print(np.max(np.abs(padt1-padt2)))

de=ea[:,None]*ea[None,:]
print(padt.shape)
exit()

a=2
d=4
time1 = time.time()
p24t = -4j*eta*wk[kidx,None,None,None]*Ctilde[kidx,a,:,None,None]*Ptilde[None,:,:,None]*Ctilde.conj()[kidx,None,d,:,None]*np.exp(-1j*(Ea[None,:,None,None]-Ea.conj()[None,None,:,None])*t[None,None,None,:])/((ea[a]-ea.conj()[d]) - (Ea[None,:,None,None]-Ea.conj()[None,None,:,None]))
time2 = time.time()
print(p24t.shape)
print(f"time = {time2 - time1}", "s")
print(p24t.nbytes / (1024**2 * 1000), "GB")

res = p24t.sum(axis=(0,1,2))
print(res.shape)

exit()

print(np.argwhere(np.abs(Ptilde[:,0]) >= 0.0001)[:,0])


padt = np.zeros((ndvr, ndvr, tpts), dtype=np.complex128)
for c in range(nbo):
    print(f"c = {c}")
    for b in np.argwhere(np.abs(Ptilde[:,c]) >= 0.0001)[:,0]:
        print(f"b = {b}")
        for k in np.argwhere(np.abs(wk)>=0.0001)[:,0]:
            print(f"k = {k}")
            padt += -4j*eta*wk[k]*Ctilde[k,:,b,None,None]*Ptilde[b,c]*Ctilde.conj()[None,k,:,c,None]*np.exp(-1j*(Ea[b]-Ea[c].conj())*t[None,None,:])/((ea[:,None,None]-ea.conj()[None,:,None]) - (Ea[b]-Ea[c].conj()))

print(padt.shape)
exit()


#Ctest = Ctilde.conj().swapaxes(1,2)


print(Ctilde.shape)
exit()

#print(np.sum(Ct0.conj()*Ct0))
print(Pt0)
#print(np.linalg.trace(Pt0))
#print(Pt0.shape)






#params = {"aR": 0.0,"bR": 0.0001,"DA" : 1.0,"bA": 0.25,"DB" : 0.8,"bB": 1.0,"aee": 0.0,"bee": 0.0001}
#model = TEGD(a=-196.7/2.0, b=196.7/2.0, N=400, bounds="(-inf,inf)", spin="singlet", model_params=params)
#Ct0 = model.wfn_to_vec(S12)





Ct0 = Ct0[:nbo]

#print(np.matmul(Ct0.conj().T,Ct0))
print(np.sum(Ct0.conj()*Ct0))
Pt0 = Ct0[:,None] * Ct0[None,:]
print(Pt0.shape)
exit()

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


