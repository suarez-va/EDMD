import numpy as np
import matplotlib.pyplot as plt

Er = np.loadtxt('Er.dat')
Gam = np.loadtxt('Gam.dat')
Cijn = np.load('Cijn.npy')
#Cijn = np.loadtxt('Cijn.dat', dtype=np.complex128)
#ndvr2, nfci = Cijn.shape
#ndvr = int(np.sqrt(ndvr2))
#Cijn = Cijn.reshape(ndvr, ndvr, nfci)

#print(Er[16])
#print(Gam[16])

npts=250
plt.figure()
for i in range(npts):
    plt.text(Er[i], Gam[i], str(i), ha='center', va='center')
plt.plot(Er[:npts],Gam[:npts],'ro')
#plt.xlim([0.0,10.0])
plt.ylim([-0.1,0.0])
#plt.ylim([-0.03,0.0])
plt.savefig("Eeta.png")
exit()


