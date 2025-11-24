import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from grid_utils.colbert_miller_dvr import dvr_p, dvr_T, dvr_xn, dvr_W

m = 1.0
a = 0.0
b = 25.
N = 400

#rmat = dvr_xn(1, a, b, N, "(-inf,inf)")
#Tmat = dvr_T(m, a, b, N, "(-inf,inf)")
#Wmat = dvr_W(a, b, N, acap=-5.0, bcap=5.0, eta=7.6e-3,n=2, bounds="(-inf,inf)")
#rmat = dvr_xn(1, a, b, N, "(a,b)")
#Tmat = dvr_T(m, a, b, N, "(a,b)")
#Wmat = dvr_W(a, b, N, acap=-5.0, bcap=5.0, eta=7.6e-3,n=2, bounds="(a,b)")
rmat = dvr_xn(1, a, b, N, "(0,inf)")
Tmat = dvr_T(m, a, b, N, "(0,inf)")
Wmat = dvr_W(a, b, N, acap=-5.0, bcap=5.0, eta=7.6e-3,n=2, bounds="(0,inf)")
print(np.diagonal(Wmat))


r=np.diagonal(rmat).real
#pot=7.5*r**2*np.exp(-np.absolute(r))
pot=np.piecewise(r, [r < 0, r >= 0], [lambda r: np.full_like(r, 10000.0), lambda r: 7.5*r**2*np.exp(-np.absolute(r))])

plt.figure()
plt.plot(r,pot)
plt.savefig("pot.png")

H = Tmat + np.diag(pot) + Wmat
Eraw, Craw = np.linalg.eig(H)
idx = np.argsort(Eraw.real)
E = Eraw[idx]
C = Craw[:, idx]

Er=E[:].real
Gam=E[:].imag

plt.figure()
for i in range(N-1):
    plt.text(Er[i], Gam[i], str(i), ha='center', va='center')
#plt.plot(Er,Gam,'ro')
plt.xlim([0.0,10.0])
plt.ylim([-7.0,0.0])
#plt.ylim([-0.03,0.0])
plt.savefig("Eeta.png")

ires=np.where((Er>3.25) & (Er<3.5) & (Gam>-0.02))
assert ires[0].shape[0]==1
ires = ires[0][0]
print(f'E={Er[ires]} - i{-Gam[ires]}')

lt.figure()
plt.plot(r,np.abs(C[:,0])**2)
plt.plot(r,np.abs(C[:,1])**2)
plt.plot(r,np.abs(C[:,2])**2)
plt.plot(r,np.abs(C[:,ires])**2,color='k')
plt.savefig("wfn.png")

answer=3.42639031-1j*0.01277448
etapt = 81
delta = 1.16
eta_ar = 1.0e-5 * (delta**np.arange(etapt)-1)/(delta-1)
print(eta_ar)
Er_ar = np.zeros((etapt))
Gam_ar = np.zeros((etapt))
for i in range(etapt):
    Wmat = dvr_W(a, b, N, acap=-5.0, bcap=8.0, eta=eta_ar[i],n=2, bounds="(0,inf)")
    H = Tmat + np.diag(pot) + Wmat
    Eraw, Craw = np.linalg.eig(H)
    idx = np.argsort(Eraw.real)
    E = Eraw[idx]
    C = Craw[:, idx]
    Er=E[:].real
    Gam=E[:].imag
    ires=np.where((Er>3.25) & (Er<3.5) & (Gam>-0.02))
    #print(ires)
    assert ires[0].shape[0]==1
    ires = ires[0][0]
    Er_ar[i]=E[ires].real
    Gam_ar[i]=E[ires].imag

plt.figure()
plt.plot(Er_ar,Gam_ar, 'ro')
plt.plot(answer.real, answer.imag, 'x',color='k', markersize='10')
#plt.xlim([float(np.min(Er_ar)),float(np.max(Er_ar))])
#plt.xlim([3.1,3.7])
#plt.ylim([-0.017,0.000])
#plt.ylim([-0.03,0.0])
plt.savefig("traj.png")





exit()



plt.figure()
plt.plot(r,np.abs(C[:,0])**2)
plt.plot(r,np.abs(C[:,1])**2)
plt.plot(r,np.abs(C[:,2])**2)
plt.plot(r,np.abs(C[:,8])**2)
plt.plot(r,np.abs(C[:,21])**2,color='k')
plt.plot(r,np.abs(C[:,28])**2,color='violet')
#plt.plot(r,np.abs(C[:,3])**2)
#plt.plot(r,np.abs(C[:,4])**2)
#plt.plot(r,np.abs(C[:,5])**2)
#plt.plot(r,np.abs(C[:,6])**2)
##plt.xlim([-20,20])
#plt.xlim([a,b])
plt.savefig("wfn.png")

m = 1.0
a = 0.0
b = 100.
N = 900

rmat = dvr_xn(1, a, b, N, "(-inf,inf)")
Tmat = dvr_T(m, a, b, N, "(-inf,inf)")
Wmat = dvr_W(a, b, N, acap=-1.0, bcap=25.0, eta=1.0e-3,n=2, bounds="(-inf,inf)")

r=np.diagonal(rmat).real
pot=np.piecewise(r, [r < 0, r >= 0], [lambda r: np.full_like(r, 1000.0), lambda r: 7.5*r**2*np.exp(-np.absolute(r))])

H = Tmat + np.diag(pot) + Wmat
Eraw, Craw = np.linalg.eig(H)
idx = np.argsort(Eraw.real)
E = Eraw[idx]
C = Craw[:, idx]

Er=E[:].real
Gam=E[:].imag

plt.figure()
for i in range(N-1):
    plt.text(Er[i], Gam[i], str(i), ha='center', va='center')
#plt.plot(Er,Gam,'ro')
plt.xlim([0.0,10.0])
plt.ylim([-7.0,0.0])
#plt.ylim([-0.03,0.0])
plt.savefig("Eeta2.png")

residx = 77
plt.figure()
plt.plot(r,np.abs(C[:,residx])**2,color='k')
plt.savefig("wfn2.png")

#etapt = 100
#delta = 1.13
#eta_ar = 1e-5 * (delta**np.arange(etapt)-1)/(delta-1)
etapt = 69
delta = 1.14
eta_ar = 2.0e-8 * (delta**np.arange(etapt)-1)/(delta-1)
print(eta_ar)
Er_ar = np.zeros((etapt))
Gam_ar = np.zeros((etapt))
for i in range(etapt):
    Wmat = dvr_W(a, b, N, acap=-1.0, bcap=25.0, eta=eta_ar[i],n=2, bounds="(-inf,inf)")
    H = Tmat + np.diag(pot) + Wmat
    Eraw, Craw = np.linalg.eig(H)
    idx = np.argsort(Eraw.real)
    E = Eraw[idx]
    C = Craw[:, idx]
    Er_ar[i]=E[residx].real
    Gam_ar[i]=E[residx].imag

plt.figure()
plt.plot(Er_ar,Gam_ar, 'ro')
plt.xlim([float(np.min(Er_ar)),float(np.max(Er_ar))])
#plt.xlim([3.1,3.7])
#plt.ylim([-0.017,0.000])
plt.ylim([-0.03,0.0])
plt.savefig("traj.png")


#
#print(ER_ar[0,:])
##print(ER_ar[-1,:])
##print(Gam_ar)
#for i in range(nsta):
#    pass
#    #plt.plot(eta_ar,Gam_ar[:,i])
#    #plt.plot(ER_ar[i,:] - ER_ar[0,:], Gam_ar[i,:])
#    #plt.plot(ER_ar[:,i] - ER_ar[0,i], Gam_ar[:,i])
#    #plt.plot(Gam_ar[i,:], ER_ar[i,:] - ER_ar[0,:])
#
#j=4
#plt.plot(ER_ar[:,j], Gam_ar[:,j], 'ro')
#plt.savefig("cap.png")


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
#plt.plot(x,pot)
#plt.axhline(y=E[0].real,xmin=a, xmax=b)
#plt.axhline(y=E[1].real,xmin=a, xmax=b)
#plt.axhline(y=E[2].real,xmin=a, xmax=b)
#plt.axhline(y=E[3].real,xmin=a, xmax=b)
#plt.axhline(y=E[4].real,xmin=a, xmax=b)
#plt.axhline(y=E[5].real,xmin=a, xmax=b)
#plt.xlim([-10,10])
##plt.xlim([a,b])
#plt.savefig("ene.png")


