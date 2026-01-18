import os
import numpy as np
import matplotlib.pyplot as plt

def get_data(nbo:int = 250):
    if not os.path.exists("Rk"):
        print("Missing grid data directory Rk")
        exit()
    kpts = int(os.popen("wc -l < Rk/Rk.dat").read().strip()) - 1
    Rk = np.loadtxt("Rk/Rk.dat", dtype=np.float64)

    sub_dir_0 = os.path.join("Rk", f"R0")
    #eigspec_0 = np.load(os.path.join(sub_dir_0, "eigspec.npz"))
    eigspec_0 = np.load(os.path.join(sub_dir_0, "eigspec2.npz"))
    xi = eigspec_0['xi']; En_0 = eigspec_0['En'][:nbo]; Cijn_0 = eigspec_0['Cijn'][:,:,:nbo]; d1En_0 = eigspec_0['d1En'][:nbo]; d1Hnm_0 = eigspec_0['d1Hnm'][:nbo,:nbo]; nac1_0 = eigspec_0['nac1'][:nbo,:nbo]; d2En_0 = eigspec_0['d2En'][:nbo]
    nbo = Cijn_0.shape[2]
    Enk = np.zeros((nbo, kpts+1), dtype=np.float64)
    d1Enk = np.zeros((nbo, kpts+1), dtype=np.float64)
    nac1_12 = np.zeros((kpts+1), dtype=np.complex128)
    d2Enk = np.zeros((nbo, kpts+1), dtype=np.float64)
    Enk[:,0] = En_0
    d1Enk[:,0] = d1En_0
    nac1_12[0] = nac1_0[12,13]
    d2Enk[:,0] = d2En_0

    for k in range(1, kpts + 1):
        sub_dir_k = os.path.join("Rk", f"R{k}")
        #eigspec_k = np.load(os.path.join(sub_dir_k, "eigspec.npz"))
        eigspec_k = np.load(os.path.join(sub_dir_k, "eigspec2.npz"))
        En_k = eigspec_k['En'][:nbo]; Cijn_k = eigspec_k['Cijn'][:,:,:nbo]; d1En_k = eigspec_k['d1En'][:nbo]; d1Hnm_k = eigspec_k['d1Hnm'][:nbo,:nbo]; nac1_k = eigspec_k['nac1'][:nbo,:nbo]; d2En_k = eigspec_k['d2En'][:nbo]
        Enk[:,k] = En_k
        d1Enk[:,k] = d1En_k
        nac1_12[k] = nac1_k[12,13]
        d2Enk[:,k] = d2En_k

    return Rk, Enk, d1Enk, nac1_12, d2Enk


nbo = 50

Rk, Enk, d1Enk, nac1_12, d2Enk = get_data()

print(np.where((Enk[:,0]>=-0.314) & (Enk[:,0]<=-0.312)))

## REMINDER TO ADD THIS SORTING TO THE MAIN ELECTRONIC STRUCTURE CALL!!!
#for k in range(Rk.shape[0]):
#    idx = np.argsort(Enk[:,k])
#    Enk[:,k] = Enk[idx,k]
#    d1Enk[:,k] = d1Enk[idx,k]
#    d2Enk[:,k] = d2Enk[idx,k]

#print(Enk[:,12]-Enk[:,12])

plt.figure()
plt.title(f"NAC")
plt.xlabel("R (a.u.)")
plt.ylabel("NAC (a.u.)")
#plt.plot(Rk, nac1_12.real, 'o', markersize=0.2, color='r')
#plt.plot(Rk, nac1_12.imag, 'o', markersize=0.2, color='b')
#plt.plot(Rk, np.absolute(nac1_12), 'o', markersize=0.2, color='k')
plt.plot(Rk, nac1_12.real, linewidth='1.0', color='r')
plt.plot(Rk, nac1_12.imag, linewidth='1.0', color='b')
plt.plot(Rk, np.abs(nac1_12), linewidth='1.0', color='k')
#for n in range(nbo):
    #plt.plot(Rk, d1Enk[n,:], linewidth='1.0')
    #plt.plot(Rk, d1Enk[n,:], 'o', markersize=0.2)

#plt.ylim([-0.4,0.0])
plt.xlim([4.19,4.21])
#plt.ylim([-0.314,-0.312])
#plt.ylim([-0.028,-0.019])
plt.savefig("nac1.png")
plt.close()

#eigspecgrid = np.load("eigspecgrid.npz")
#Rk = eigspecgrid["Rk"]
#Enk = eigspecgrid["Enk"]

plt.rcParams.update({
    'figure.figsize': (4.0, 3.0),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'lines.linewidth': 2,
    'axes.linewidth': 3,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.right': True,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.frameon': False,
})

plt.figure()
plt.title(f"Hdvr En")
plt.xlabel("R (a.u.)")
plt.ylabel("En (a.u.)")

kidx = 11
E12k = Enk[12,kidx]+d1Enk[12,kidx]*(Rk - Rk[kidx])+0.5*d2Enk[12,kidx]*(Rk-Rk[kidx])**2
E13k = Enk[13,kidx]+d1Enk[13,kidx]*(Rk - Rk[kidx])+0.5*d2Enk[13,kidx]*(Rk-Rk[kidx])**2
plt.plot(Rk[kidx], Enk[12,kidx], 'ro', markersize=1.5)
plt.plot(Rk[kidx], Enk[13,kidx], 'ro', markersize=1.5)
plt.plot(Rk, E12k, linewidth='0.3')
plt.plot(Rk, E13k, linewidth='0.3')
for n in range(nbo):
    #plt.plot(Rk, Enk[n,:], linewidth='1.0')
    plt.plot(Rk, Enk[n,:], 'o', markersize=0.7)

#plt.ylim([-0.4,0.0])
plt.xlim([4.19,4.21])
plt.ylim([-0.314,-0.312])
#plt.ylim([-0.028,-0.019])
plt.savefig("EnvsR.png")
