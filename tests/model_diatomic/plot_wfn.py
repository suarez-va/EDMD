from grid_utils.generate_grid import sort_eta_data, get_eta_data
import numpy as np
import matplotlib.pyplot as plt

#eta_ar, Er_ar, Gam_ar = get_eta_data(16)
#eta_ar, Er_ar, Gam_ar = get_eta_data(38)

idx = 16

eigspec = np.load('eigspec.npz')
xi = eigspec['xi']
En = eigspec['En']
Cijn = eigspec['Cijn']

Pi = 2 * np.sum(np.absolute(Cijn[:,:,idx])**2, axis=1)
#Pi = 2 * np.sum(np.absolute(Cijn[:,:,0])**2, axis=1)
plt.figure()
plt.plot(xi, Pi, color='k')
plt.savefig("rho.png")

plt.figure()
#plt.contourf(np.absolute(Cijn[:,:,16]), levels=250)
plt.contourf(np.absolute(Cijn[:,:,idx]), levels=250)
plt.colorbar()
plt.savefig("wfn.png")

exit()

xpts = Cijn.shape[0]
Cij_approx = np.zeros((xpts,xpts), dtype=np.complex128)
#Crnm0 = np.loadtxt("etan/eta0/Crnm.dat", dtype=np.complex128)
Crnm = np.loadtxt("etan/eta20/Crnm.dat", dtype=np.complex128)

for n in range(100):
    Cij_approx[:,:] += Crnm[n,idx] * Cijn[:,:,n]

plt.figure()
#plt.contourf(np.absolute(Cijn[:,:,16]), levels=250)
plt.contourf(np.absolute(Cij_approx), levels=250)
plt.colorbar()
plt.savefig("wfn_approx.png")


