import numpy as np
import time
from models.two_electron_screened_diatomic import TESD, generate_Hele, generate_dHele, generate_dipole, generate_cap, Vnuc
import matplotlib.pyplot as plt

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
    "xmax": 50.0,
    "xpts": 125,
    "xcap": 17.5,
    "etacap": 1.0,
    "ncap": 3,
    "spin": "triplet"
}

model = TESD(params)

#ovlp = model.ovlp(1.0)
##print(np.max(np.absolute(ovlp - np.eye(ovlp.shape[0]))))
#print(np.diag(ovlp))
#exit()

Ri = 2.5
start = time.time()
#Hi = model.hcore(Ri)
Hi = model.Hele(Ri)
stop = time.time()
print(f'H build: {stop-start}')
start = time.time()
Ei, Ci = np.linalg.eigh(Hi)
stop = time.time()
print(f'eigh: {stop-start}')

nsta=7750
for i in range(nsta):
    plt.axhline(y=Ei[i], xmin=-1, xmax=1)
plt.savefig("Ci.png")
exit()

#print(Ei[1:] - Ei[:-1])
#plt.plot(np.absolute(Ci[:,0])**2)
#plt.plot(np.absolute(Ci[:,1])**2)
#plt.plot(np.absolute(Ci[:,2])**2)
#plt.plot(np.absolute(Ci[:,3])**2)
#plt.plot(np.absolute(Ci[:,4])**2)
#plt.plot(np.absolute(Ci[:,5])**2)
#plt.plot(np.absolute(Ci[:,6])**2)
#plt.plot(np.absolute(Ci[:,7])**2)
##plt.plot(np.absolute(Ci[:,8])**2)
##plt.plot(np.absolute(Ci[:,9])**2)
##plt.plot(np.absolute(Ci[:,10])**2)
##plt.plot(np.absolute(Ci[:,11])**2)
#plt.savefig("Ci.png")
#exit()

Rpts=24
nsta=13
#R=np.linspace(0.4, 15.0, Rpts)
R=np.linspace(0.4, 8.0, Rpts)
ER=np.zeros((Rpts,nsta))
Gamma=np.zeros((Rpts,nsta))
for i in range(Rpts):
    print(i)
    Ri = R[i]
    #hcore = model.hcore(Ri)
    start = time.time()
    Hele = model.Hele(Ri)
    stop = time.time()
    print(f'Hele build: {stop-start}')
    start = time.time()
    Ei, Ci = np.linalg.eigh(Hele)
    stop = time.time()
    print(f'eigh: {stop-start}')
    ER[i,:]=Ei[:nsta] + model.Vnuc(Ri)
    #ER[i,:]=Ei[:nsta].real
    #Gamma[i,:]=Ei[:nsta].imag

np.savetxt('R.dat', R)
np.savetxt('ER.dat', ER)
np.savetxt('Gamma.dat', Gamma)


#Rpts=7
#nsta=12
#R=np.linspace(1.0, 4.0, Rpts)
##E=np.zeros((Rpts,nsta), dtype=np.complex128)
#ER=np.zeros((Rpts,nsta))
#Gamma=np.zeros((Rpts,nsta))
#for i in range(Rpts):
#    print(i)
#    Ri = R[i]
#    Hele = generate_Hele(Ri, params)
#    #Ei, Ci = np.linalg.eig(Hele)
#    Ei, Ci = np.linalg.eigh(Hele)
#    ER[i,:]=Ei[:nsta] + Vnuc(Ri, params)
#    #ER[i,:]=Ei[:nsta].real
#    #Gamma[i,:]=Ei[:nsta].imag
#
#np.savetxt('R.dat', R)
#np.savetxt('ER.dat', ER)
#np.savetxt('Gamma.dat', Gamma)
