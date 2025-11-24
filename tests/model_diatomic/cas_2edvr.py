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
    "xmax": 9.0,
    "ndvr": 300,
    "ncas": 111,
    "xcap": 17.5,
    "etacap": 1.0,
    "ncap": 3,
    "spin": "singlet"
}

R = 2.5
model = TESD(params)
t1=time.time()
model.solve_mos(R)
t2=time.time()
print(f'solve_mos() time = {t2-t1}')
Hprqs = model.Hprqs()
t3=time.time()
print(f'Hprqs contraction time = {t3-t2}')
H = model.Hele(R)
t4=time.time()
print(f'Hele generation time = {t4-t3}')
E, C = np.linalg.eigh(H)
t5=time.time()
print(f'Hele diagonalization time = {t5-t4}')
#print(f'single={model.ep[0]+model.ep[0]}')
#print(model.ep[0]+model.ep[1])
#print(model.ep[0]+model.ep[2])
#print(model.ep[0]+model.ep[3])
#print(model.ep[0]+model.ep[4])
#print(model.ep[0]+model.ep[5])
#print(model.ep[1]+model.ep[2])
print(E[:3])

x_ar = model.xi()
pot_ar = model.VeR(x_ar, R)
e=model.ep
c=model.cip
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

