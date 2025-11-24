import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from grid_utils.colbert_miller_dvr import dvr_p, dvr_T, dvr_xn, dvr_W

m = 1.0
a = -25.
b = 25.
N = 725

xmat = dvr_xn(1, a, b, N, "(-inf,inf)")
Tmat = dvr_T(m, a, b, N, "(-inf,inf)")

D=10.0
A=1.0
x=np.diagonal(xmat).real
pot = -D*np.exp(-A*x**2)

etapt = 100
nsta = 35
delta = 1.2
eta_ar = 1e-5 * (delta**np.arange(etapt)-1)/(delta-1)
print(eta_ar)
ER_ar = np.zeros((etapt,nsta))
Gam_ar = np.zeros((etapt,nsta))
for i in range(etapt):
    cap = dvr_W(a, b, N, -10, 10, eta_ar[i], n=4, bounds="(-inf,inf)")
    H = Tmat + np.diag(pot) + cap
    E, C = np.linalg.eig(H)
    idx = np.argsort(E.real)
    Esort = E[idx]
    Csort = C[:, idx]
    ER_ar[i,:] = Esort[:nsta].real
    Gam_ar[i,:] = Esort[:nsta].imag

print(ER_ar[0,:])
#print(ER_ar[-1,:])
#print(Gam_ar)
for i in range(nsta):
    pass
    #plt.plot(eta_ar,Gam_ar[:,i])
    #plt.plot(ER_ar[i,:] - ER_ar[0,:], Gam_ar[i,:])
    #plt.plot(ER_ar[:,i] - ER_ar[0,i], Gam_ar[:,i])
    #plt.plot(Gam_ar[i,:], ER_ar[i,:] - ER_ar[0,:])

j=4
plt.plot(ER_ar[:,j], Gam_ar[:,j], 'ro')
plt.savefig("cap.png")


#H=Tmat + np.diag(pot)
#E, C = np.linalg.eigh(H)

#plt.plot(x,np.abs(C[:,0])**2)
#plt.plot(x,np.abs(C[:,1])**2)
#plt.plot(x,np.abs(C[:,2])**2)
#plt.plot(x,np.abs(C[:,3])**2)
#plt.plot(x,np.abs(C[:,4])**2)
#plt.plot(x,np.abs(C[:,5])**2)
#plt.plot(x,np.abs(C[:,6])**2)
##plt.xlim([-20,20])
#plt.xlim([a,b])
#plt.savefig("wfn.png")

#print(pot)
#print(E[0].real)
#print(E[1].real)
#print(E[2].real)
plt.plot(x,pot)
plt.axhline(y=E[0].real,xmin=a, xmax=b)
plt.axhline(y=E[1].real,xmin=a, xmax=b)
plt.axhline(y=E[2].real,xmin=a, xmax=b)
plt.axhline(y=E[3].real,xmin=a, xmax=b)
plt.axhline(y=E[4].real,xmin=a, xmax=b)
plt.axhline(y=E[5].real,xmin=a, xmax=b)
plt.xlim([-10,10])
#plt.xlim([a,b])
plt.savefig("ene.png")


