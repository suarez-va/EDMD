import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', default=0, type=int, help='index of bo state to plot')
args = parser.parse_args()
n = args.n

if not os.path.exists("etai"):
    print("Missing grid data directory etai")
    exit()

capgrid_data = np.load('etai/fcicap_2ele_singlet_capgrid.npz')
etai = capgrid_data['etai']
Eni = capgrid_data['Eni']
ipts = etai.shape[0]
nbo = Eni.shape[0]
Ei = Eni[n,:]
#Ei[0] = Ei[1]


plt.rcParams.update({
    'figure.figsize': (6.0, 4.0),
    'figure.dpi': 150,
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

ax1.set_xlabel('Re[E(η)]')
ax1.set_ylabel('Im[E(η)]')
ax1.set_title(f"E(η) traj; n = {n}")
for i in range(ipts):
    ax1.text(Ei[i].real, Ei[i].imag, str(i), ha='center', va='center', fontsize=7)
ax1.plot(Ei.real, Ei.imag,'ro')

ax2.set_xlabel('η index')
ax2.set_ylabel('η|dE(η)/dη|')
dEi = np.zeros((ipts-1), dtype=np.complex128)
for i in range(ipts-1):
    dEi[i] = (Ei[i+1]-Ei[i])/(etai[i+1]-etai[i])
etaidEi = np.absolute(etai[:-1] * dEi)
ax2.plot(np.arange(ipts-1), etaidEi, color='k')
plt.subplots_adjust(hspace=0.2, left=0.23, right=0.95, top=0.90, bottom=0.20)


#print(El[40])
#print(El[50])
#print(El[60])
#print(El[70])
#print(El[80])
#print(El[90])
imin = 20 + np.argmin(etaidEi[20:100])
print(imin)
print(Ei[imin])
Gamma = -2*Ei[imin].imag
print(f"Γ = {27211.386 * Gamma} meV")
print(f"η = {etai[imin]}")

plt.show()

#plt.savefig("El.png")

#fig, ax = plt.subplots()
#ax.set_xlim(-1.0, 0.0)
#ax.set_ylim(-0.005, 0.001)
#ax.axhline(y=0.0, linestyle='--', color='k')
#ax.set_xlabel('Re[E(η)]')
#ax.set_ylabel('Im[E(η)]')
#ax.set_title("E(η) traj; n = {n}")
#ax.plot([],[],'ro', markersize=5)
#texts = [ax.text(0, 0, '', ha='center', va='center', fontsize=9) for _ in range(nbo)]
#
#def update(frame):
#    title.set_text(f"η = {np.round(etal[frame], 8)}")
#    plot.set_data(Enl[:,frame].real, Enl[:,frame].imag)
#    for n, text in enumerate(texts):
#        text.set_position((Enl[n, frame].real, Enl[n, frame].imag))
#        text.set_text(str(n))
#    return [plot, title] + texts
#
#ani = FuncAnimation(fig, func=update, frames=lpts, blit=False)
#
#plt.show()

#fig, ax = plt.subplots()
#ax.set_xlim(-1.0, 0.0)
#ax.set_ylim(-0.005, 0.0)
#for l, eta in enumerate(etal):
#    sub_dir = f"etal/eta{l}"
#    capspec = np.load(sub_dir + "/capspec.npz")
#    En = capspec["En"] ; Cnml = capspec["Cnml"]; Cnmr = capspec["Cnmr"]
#    for n, E in enumerate(En):
#        plt.text(E.real, E.imag, str(n), ha='center', va='center')
#    plt.plot(En.real,En.imag,'ro')
##plt.xlim([0.0,10.0])
#plt.ylim([-0.005,0.0])
#plt.savefig("Eeta.png")
#exit()

