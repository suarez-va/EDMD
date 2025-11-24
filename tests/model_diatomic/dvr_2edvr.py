#import os
#os.environ["MKL_NUM_THREADS"] = "4"
#os.environ["OMP_NUM_THREADS"] = "4"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
#np.__config__.show()
from grid_utils.colbert_miller_dvr import dvr_p, dvr_T, dvr_xn, dvr_W
from models.two_electron_screened_diatomic import TESD, generate_Hele, generate_dHele, generate_dipole, generate_cap, Vnuc
import matplotlib.pyplot as plt
import time
from threadpoolctl import threadpool_limits

#N=4096
#M=np.random.rand(N,N)+1j*np.random.rand(N,N)
#ti = time.time()
#with threadpool_limits(limits=4):
#    e, V = np.linalg.eig(M)
#tf = time.time()
#print(f'total time = {tf-ti}')
#exit()


#def map_ci_dvr(self, params):
#    # SINGLET generate index mapping between single particle grid points and DVR slater determinants
#    ndvr = self.params["ndvr"]
#    map = np.full((ndvr, ndvr), -1)
#    k = 0
#    for i in range(xpts):
#        for j in range(i,xpts):
#            map[i,j] = k
#            k += 1
#    return map


#def Hele(params, R, h):
#    xmax = params["xmax"]
#    ndvr = params["ndvr"]
#
#    dx = (2 * xmax) / (ndvr - 1)
#    # generate hcore using Colbert-Miller syle DVR for kinetic energy
#     # generate full CI Hele matrix
#    map = np.full((ndvr, ndvr), -1)
#    k = 0
#    for i in range(ndvr):
#        for j in range(i,ndvr):
#            map[i,j] = k
#            k += 1
#    nstates = int((ndvr + 1) * ndvr / 2)
#    H = np.zeros((nstates, nstates), dtype=complex)
#    for iket in range(ndvr):
#        for jket in range(iket,ndvr):
#            for ibra in range(ndvr):
#                for jbra in range(ibra,ndvr):
#                    xi = -xmax + dx * iket; xj = -xmax + dx * jket; xarg = (xi - xj)**2
#                    dii = 1 if iket == ibra else 0
#                    djj = 1 if jket == jbra else 0
#                    dij = 1 if iket == jbra else 0
#                    dji = 1 if jket == ibra else 0
#                    H[map[iket,jket],map[ibra,jbra]] += dii*h[jket,jbra] + djj*h[iket,ibra] + dij*h[jket,ibra] + dji*h[iket,jbra]
#                    #H[map[iket,jket],map[ibra,jbra]] += (dii*djj+dij*dji)*np.exp(-aee*xarg)/np.sqrt(xarg+bee)
#                    if dii == 1 and djj == 1 and dij == 1:
#                        H[map[iket,jket],map[ibra,jbra]] *= 0.5
#    return H



params = {
    "aee": 0.02,
    "bee": 0.25,
    "aR": 0.01205,
    "bR": 0.01,
    "aAe": 0.0102,
    "bAe": 0.655,
    "aBe": 0.0139,
    "bBe": 0.473,
    "mA": 36443.98900696,
    "mB": 7294.29954142,
    # max=20 then ndvr=150, but do more honestly if you want many digits
    "xmax": 15.0,
    "ndvr": 100,
    "ncas": 100,
    "xcap": 10.0,
    "eta": 0.032,
    "ncap": 4,
    "spin": "singlet"
}

R = 2.5
model = TESD(params)

x_ar = model.xi()
pot_ar = model.VeR(x_ar, R)
model.solve_mos(R)
e=model.ep
print(e.dtype)
c=model.cip

er = e[:].real
gam = e[:].imag

npts=50
plt.figure()
for i in range(npts):
    plt.text(er[i], gam[i], str(i), ha='center', va='center')
plt.plot(er[:npts],gam[:npts],'ro')
plt.savefig("eeta.png")

#print(e[0] + e[0])
#print(e[0] + e[1])
#print(e[0] + e[2])
#print(f'1e norm: {np.sum(np.abs(c[:,0])**2)}')
#print(f'1e norm: {np.sum(c[:,3].conj() * c[:,3])}')
#print(f'1e orth: {np.sum(c[:,3].conj() * c[:,2])}')
#print(e[0] + e[3])

model.solve_wfn(R)
E = model.En
C = model.Cijn

Er = E[:].real
Gam = E[:].imag

npts=500
plt.figure()
for i in range(npts):
    plt.text(Er[i], Gam[i], str(i), ha='center', va='center')
plt.plot(Er[:npts],Gam[:npts],'ro')
#plt.xlim([0.0,10.0])
plt.ylim([-0.1,0.0])
#plt.ylim([-0.03,0.0])
plt.savefig("Eeta.png")
exit()



#FS=1/np.sqrt(2) * (np.einsum('i,j->ij',c[:,0],c[:,1]) + np.einsum('i,j->ij',c[:,1],c[:,0]))
#FT=1/np.sqrt(2) * (np.einsum('i,j->ij',c[:,0],c[:,1]) - np.einsum('i,j->ij',c[:,1],c[:,0]))
#print(FS)
#print(f'2e approx norm: {np.sum(np.sum(FS.conj() * FS, axis=1), axis=0)}')
#print(FS[0,0])
#print(FS[0,1])
#print(FS[1,0])
#print(FT)
#print(f'2e approx norm: {np.sum(np.sum(FT.conj() * FT, axis=1), axis=0)}')
#print(FT[0,0])
#print(FT[0,20])
#print(FT[20,0])

model.solve_wfn(R)
E = model.En
C = model.Cijn

print("Energy:")
print(e[0] + e[1])
#print(E[1])
print(E[0])

print(f'2e norm: {np.sum(np.sum(C[:,:,3].conj() * C[:,:,3], axis=1), axis=0)}')
print(f'2e norm: {np.sum(np.sum(C[:,:,0].conj() * C[:,:,1], axis=1), axis=0)}')
print(f'2e norm: {np.sum(np.sum(C[:,:,0].conj() * C[:,:,0], axis=1), axis=0)}')

#C1 = C[:,:,1]
#print(C1[0,0])
#print(C1[0,1])
#print(C1[1,0])
#print(f'2e norm: {np.sum(np.sum(C1.conj() * C1, axis=1), axis=0)}')
#C0 = C[:,:,0]
#print(C0[0,0])
#print(C0[0,20])
#print(C0[20,0])
#print(f'2e norm: {np.sum(np.sum(C0.conj() * C0, axis=1), axis=0)}')
#print(f'2e norm: {np.sum(np.sum(C[:,:,3].conj() * C[:,:,3], axis=1), axis=0)}')
#exit()

#Cdiag = np.diagonal(model.Cijn, axis1=0, axis2=1).T
#Cdiag = np.eye(model.ndvr)[:, :, None] * Cdiag[:,None,:]

#Cdiag = np.eye(model.ndvr)[:,:,None] * np.diagonal(model.Cijn,axis1=0,axis2=1).T[None,:,:]

#print(np.diag(Cdiag))

#Cdiag = np.einsum('ij,jn->ijn', np.eye(model.ndvr), Cdiag)
#Cdiag = Cdiag[:,None,:] * np.eye(Cdiag.shape[0])[:, :, None]

#C0 = model.Cijn[:,:,0]
#print(C0.shape)
#print(C0[40,42])
#print(C0[42,40])
#print(C0[41,41])
#
#print(C0)

plt.figure()
plt.contour(np.absolute(C[:,:,0])**2, levels=50)
plt.title("Contour Plot")
plt.savefig("wfn.png")

exit()

print(C0.shape)
print(model.Cijn.shape)
print(Cdiag.shape)

print(C0)
print(Cdiag[:,:,0])
print(np.diag(Cdiag[:,:,0]))

print(model.En)
exit()

#mij,mkl,mikjl = model.ci_map()
#print(np.max(mij-model.map_ij))
#print(np.max(mkl-model.map_kl))
#print(np.max(mikjl-model.map_ikjl))
#
#exit()
#ij, kl = np.triu_indices(model.ndvr, k=0)
#
#map_ij = ij.copy()
#map_kl = kl.copy()
#map_ikjl = np.full((model.ndvr, model.ndvr), -1, dtype=int)
#map_ikjl[ij, kl] = np.arange(model.nfci)
#
#print(np.max(mij-map_ij))
#print(np.max(mkl-map_kl))
#print(np.max(mikjl-map_ikjl))
#
#exit()

t1=time.time()
Hikjl = model.Hikjl(R)
t2=time.time()
print(f'Hikjl contraction time = {t2-t1}')
Hdvr = model.Hdvr(R)
t3=time.time()
print(f'Hdvr generation time = {t3-t2}')
#print(np.diagonal(Hdvr))
#print(np.max(Hdvr-np.diag(np.diagonal(Hdvr))))
#exit()
E, C = np.linalg.eigh(Hdvr)
t4=time.time()
print(f'Hdvr diagonalization time = {t4-t3}')
#print(f'single={model.ep[0]+model.ep[0]}')
#print(model.ep[0]+model.ep[1])
#print(model.ep[0]+model.ep[2])
#print(model.ep[0]+model.ep[3])
#print(model.ep[0]+model.ep[4])
#print(model.ep[0]+model.ep[5])
#print(model.ep[1]+model.ep[2])
print(E[:3])
exit()

#hij = model.hij(R)
#Hold = Hele(params, R, hij)
#E, C = np.linalg.eigh(Hold)
#print(E[:3])

Hcas = model.Hele(R)
E, C = np.linalg.eigh(Hcas)
print(E[:3])

dij = np.eye(params["ndvr"])
#Hikjl *= 1 - 0.5 * dij[:,None,:,None] * dij[None,:,None,:] * dij[:,None,None,:] * dij[None,:,:,None]
Hpkjl=np.einsum('ip,ikjl->pkjl',c.conj(),Hikjl)
Hpkql=np.einsum('jq,pkjl->pkql',c,Hpkjl)
Hprql=np.einsum('kr,pkql->prql',c.conj(),Hpkql)
Hprqs_rot=np.einsum('ls,prql->prqs',c,Hprql)

#kHprqs_rot=np.einsum('ls,prql->prqs',c,Hprql)

dpq = np.eye(params["ncas"])
Hprqs=model.Hprqs()
#Hprqs *= 1 - 0.5 * dpq[:,None,:,None] * dpq[None,:,None,:] * dpq[:,None,None,:] * dpq[None,:,:,None]
print(f'Correct Hpqrs:')
#print(Hprqs)
print(f'Incorrect Hpqrs:')
#print(Hprqs_rot)
print(np.max(Hprqs-Hprqs_rot))

exit()
print("NEXT TEST, AM I MESSING WITH OFF DIAGONALS?")

Iikjl = np.ones(Hikjl.shape)
print(Iikjl)
Iikjl *= 1 - 0.5 * dij[:,None,:,None] * dij[None,:,None,:] * dij[:,None,None,:] * dij[None,:,:,None]
print(np.diagonal(Iikjl))
print(np.argwhere(Iikjl))

exit()

#u=np.linalg.inv(c)
#Hirqs=np.einsum('pi,prqs->irqs',u.conj(),Hprqs)
#Hirqs=np.einsum('pi,irqs->irqs',u.conj(),Hprqs)
#
#
#Hpkql=np.einsum('jq,pkjl->pkql',c,Hpkjl)
#Hprql=np.einsum('rk,pkql->prql',c.conj(),Hpkql)
#Hprqs_rot=np.einsum('ls,prql->prqs',c,Hprql)




exit()

Nsta = 23
Nnorm = 5
plt.figure()
plt.plot(x_ar,pot_ar, color='k', linewidth=1.5)
for i in range(Nsta):
    plt.axhline(e[i], color='k', linewidth=0.7)
    plt.plot(x_ar,Nnorm * (i+1)**(1.1) * np.abs(c[:,i])**2+e[i], linewidth=1.2)
plt.xlim([-25,25])
plt.savefig("wfn.png")



#Hprqs2 = model.Hprqs2()
#t4=time.time()
#print(f'old Hprqs contraction time = {t4-t3}')
#print(Hprqs2)
#print(np.max(np.absolute(Hprqs-Hprqs2)))

#x_ar = model.xi()
#Vik = model.Vee(x_ar[:,None], x_ar[None,:])
#print(x_ar[:,None])
#print(Vik)
exit()



xpts = params["xpts"]
ncas = params["ncas"]
x_ar = model.xi()
Vik = model.Vee(x_ar[:,None], x_ar[None,:])

t1=time.time()
Cipq = np.einsum('ip,iq->ipq', model.cip[:,:ncas].conj(), model.cip[:,:ncas])
t2=time.time()
print(f'Cipq contraction time = {t2-t1}')
Cin = Cipq.reshape(xpts,ncas**2)
Vin = np.matmul(Vik, Cin)
Virs2 = Vin.reshape(xpts,ncas,ncas)
t3=time.time()
print(f'Vin contraction time = {t3-t2}')
Virs = np.einsum('ik,krs->irs',Vik, Cipq)
t4=time.time()
print(f'Virs contraction time = {t4-t3}')
print(Virs2)
print(np.max(np.absolute(Virs2-Virs)))
exit()
Vprqs = np.einsum('ipq,irs->prqs', Cipq, Virs)
#Vprqs = np.einsum('ip,iq,irs->prqs',C[:,:cas].conj(), C[:,:cas], Virs)
t4=time.time()
print(f'Vprqs contraction time = {t4-t3}')


exit()
print(f'solve_mos() time = {t2-t1}')
Hprqs = model.Hprqs()
t3=time.time()
print(f'Hprqs contraction time = {t3-t2}')
print(Hprqs.shape)
exit()

hcore = model.hcore(R)
E, C = np.linalg.eigh(hcore)
#ER[i,:]=Ei[:nsta] + model.Vnuc(Ri)

p=0
q=0
r=0
s=0
aee = params["aee"]
bee = params["bee"]
xmax = params["xmax"]
xpts = params["xpts"]
dx = (2 * xmax) / (xpts - 1)
x_ar=np.zeros((xpts))
for j in range(xpts):
    xj = -xmax + dx * j
    x_ar[j] = xj
xi_ar=x_ar[:,None]
xk_ar=x_ar[None,:]
#print(xi_ar.shape); print(xi_ar)
#print(xk_ar.shape); print(xk_ar)
xarg = (xi_ar - xk_ar)**2
Vik = np.exp(-aee*xarg)/np.sqrt(xarg+bee)
#print(Vik.shape); print(Vik)
#print(np.einsum('i,ij,j',C[:,p].conj(), hcore, C[:,q]))
#print(np.einsum('k,k',C[:,r].conj(), C[:,s]))
#Vpqrs = np.einsum('i,i,i',C[:,p].conj(), C[:,q], np.einsum('ik,k,k->i',Vik, C[:,r].conj(), C[:,s]))
#print(f'p={p},q={q},r={r},s={s},Vpqrs={Vpqrs}')

plt.figure()
plt.plot(x_ar,np.abs(C[:,0])**2)
plt.plot(x_ar,np.abs(C[:,1])**2)
plt.plot(x_ar,np.abs(C[:,2])**2)
plt.plot(x_ar,np.abs(C[:,75])**2,color='k')
plt.xlim([-25,25])
plt.savefig("wfn.png")

cas=params["cas"]
map_pq, map_rs, map_prqs = model.map_ci()
#print(map_pq[map_p[0],map_q[0]])
#print(map_pq[map_p[93],map_q[93]])
#print(map_pq[map_p[94],map_q[94]])

t1=time.time()
I = np.eye(cas)
hprqs = (E[:cas,None,None,None]+E[None,:cas,None,None])*(I[:,None,:,None]*I[None,:,None,:]-I[:,None,None,:]*I[None,:,:,None])
#h1=np.einsum('p,pq,rs->prsq',E[:cas],np.eye(cas),np.eye(cas))
t2=time.time()
print(f'hprqs contraction time = {t2-t1}')

t1=time.time()
Cipq = np.einsum('ip,iq->ipq', C[:,:cas].conj(), C[:,:cas])
t2=time.time()
print(f'Cipq contraction time = {t2-t1}')
Virs = np.einsum('ik,krs->irs',Vik, Cipq)
#Virs = np.einsum('ik,kr,ks->irs',Vik, C[:,:cas].conj(), C[:,:cas])
t3=time.time()
print(f'Virs contraction time = {t3-t2}')
Vprqs = np.einsum('ipq,irs->prqs', Cipq, Virs)
#Vprqs = np.einsum('ip,iq,irs->prqs',C[:,:cas].conj(), C[:,:cas], Virs)
t4=time.time()
print(f'Vprqs contraction time = {t4-t3}')

exit()


ti=time.time()
Cnmpq=C[:,map_pq]
Cnmrs=C[:,map_rs]
#This one is way too slow Vnm = np.einsum('in,im,ik,kn,km->nm', Cnmpq.conj(), Cnmpq, Vik, Cnmrs.conj(), Cnmrs)
#DON'T DO THIS, BREAKS COMPUTER Vinm = np.einsum('ik,kn,km->inm',Vik, C_nmrs.conj(), C_nmrs); Vnm = np.einsum('in,im,inm->nm',C_nmpq.conj(), C_nmpq, Vinm)
#Vim = np.einsum('ik,km,km->im',Vik, C_nmrs.conj(), C_nmrs)
#Vnm = np.einsum('in,in,im->nm',C_nmpq.conj(), C_nmpq, Vim)
tf=time.time()
print(f'contraction2 time = {tf-ti}')

i=7
n=3
m=5

print(Vnm[n,m])
print(Vprqs[map_pq[n],map_rs[n],map_pq[m],map_rs[m]])
exit()

Vmax=0
p=cas-1
ti=time.time()
Virs = np.einsum('ik,kr,ks->irs',Vik, C[:,:cas].conj(), C[:,:cas])
Vpqrs = np.einsum('ip,iq,irs->pqrs',C[:,:cas].conj(), C[:,:cas], Virs)
tf=time.time()
print(f'contraction1 time = {tf-ti}')
#print(np.max(Vpqrs[p,:,:,:]))
exit()
for q in range(cas):
    for r in range(cas):
        for s in range(cas):
            vpqrs = np.einsum('i,i,i',C[:,p].conj(), C[:,q], np.einsum('ik,k,k->i',Vik, C[:,r].conj(), C[:,s]))
            vpqrs = Vpqrs[p,q,r,s]
            #print(f'check if {Vpqrs[p,q,r,s]}={vpqrs}')
            if vpqrs>=Vmax:
                Vmax=vpqrs
                print(f'p={p},q={q},r={r},s={s},Vpqrs={vpqrs}')


#print(np.max(np.einsum('kr,ks->rs',C.conj(), C)-np.eye(C.shape[0])))
#h1=np.einsum('ip,ij,jq,kr,ks->pqrs',C.conj(), hcore, C, C.conj(), C)
#print(h1.shape)
#print(h1[2,2,6,6]-E[2])
#print(h1[2,1,6,6])
#print(h1[2,2,6,5])
#print(h1[1,2,3,4])
#print(h1[1,1,8,9])
#print(h1[1,1,8,8])
#print(h1[19,19,0,0])
exit()

aee = self.params["aee"]
bee = self.params["bee"]
xmax = self.params["xmax"]
xpts = self.params["xpts"]
dx = (2 * xmax) / (xpts - 1)
#xi = -xmax + dx * iket; xj = -xmax + dx * jket; xarg = (xi - xj)**2
#np.exp(-aee*xarg)/np.sqrt(xarg+bee)

print("1:")
print(generate_Hele(R, params))

print("2:")
print(generate_dipole(R, params))

print("3:")
print(generate_cap(R, params))

