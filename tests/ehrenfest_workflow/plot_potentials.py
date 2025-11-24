import matplotlib.pyplot as plt
import numpy as np

R=np.loadtxt("R.dat")
ER=np.loadtxt("ER.dat")
Gamma=np.loadtxt("Gamma.dat")

plt.plot(R, ER[:,0])
plt.plot(R, ER[:,1])
plt.plot(R, ER[:,2])
plt.plot(R, ER[:,3])
plt.plot(R, ER[:,4])
plt.plot(R, ER[:,5])
plt.ylim([-2.5,0.0])
plt.savefig("dat.png")

