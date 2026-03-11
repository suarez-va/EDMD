import numpy as np
from model_systems.models import GICD

#params = {
#    'ZA': 0.0,
#    'ZB': 0.5,
#    'aR': 0.0,
#    'bR': 0.0001,
#    'DA' : 1.0,
#    'bA': 0.25,
#    'DB' : 0.8,
#    'bB': 1.0,
#    'aee': 0.0,
#    'bee': 0.0001,
#}

#model = GICD(params)
#x_sub = 0.3
#R_sub = 2.3
#dR_sub = 0.001
#print(model.VR(R_sub))
#print(model.d1VR(R_sub))
#print(model.d2VR(R_sub))
#Ve_Rm1 = model.VeR(x_sub, R_sub - dR_sub)
#Ve_R = model.VeR(x_sub, R_sub)
#Ve_Rp1 = model.VeR(x_sub, R_sub + dR_sub)
#d1Ve_Rm1 = model.d1VeR(x_sub, R_sub - dR_sub)
#d1Ve_R = model.d1VeR(x_sub, R_sub)
#d1Ve_Rp1 = model.d1VeR(x_sub, R_sub + dR_sub)
#d2Ve_R = model.d2VeR(x_sub, R_sub)
#d2Ve_approx0 = (Ve_Rp1 - 2*Ve_R + Ve_Rm1) / (dR_sub**2)
#d2Ve_approx1 = (d1Ve_Rp1 - d1Ve_R) / dR_sub
#d2Ve_approx2 = (d1Ve_R - d1Ve_Rm1) / dR_sub
#print(f'd2VeR: 0: {d2Ve_approx0}, 1: {d2Ve_approx1}, 2: {d2Ve_approx2}, res: {d2Ve_R}')


output_file = 'fcidvr_1ele_doublet.npz'
#output_file = 'fcidvr_2ele_singlet.npz'

RI = np.loadtxt("RI/RI.dat", dtype=np.float64)
dR = (RI[-1] - RI[0]) / (RI.shape[0] - 1)

I=25
output_Rm1 = np.load(f"RI/R{I-1}/" + output_file)
output_R = np.load(f"RI/R{I}/" + output_file)
output_Rp1 = np.load(f"RI/R{I+1}/" + output_file)

En_Rm1 = output_Rm1['En']
En_R = output_R['En']
En_Rp1 = output_Rp1['En']
nbo = En_R.shape[0]

Cn_Rm1 = output_Rm1['Cn']
Cn_R = output_R['Cn']
Cn_Rp1 = output_Rp1['Cn']

d1En_R = output_R['d1En']
d2En_R = output_R['d2En']
nac01_Rm1 = output_Rm1['nac01']
nac01_R = output_R['nac01']
nac01_Rp1 = output_Rp1['nac01']
nac11_Rm1 = output_Rm1['nac11']
nac11_R = output_R['nac11']
nac11_Rp1 = output_Rp1['nac11']
nac02_Rm1 = output_Rm1['nac02']
nac02_R = output_R['nac02']
nac02_Rp1 = output_Rp1['nac02']

d1En_approx1 = (En_Rp1 - En_R) / dR
d1En_approx2 = (En_R - En_Rm1) / dR
d2En_approx0 = (En_Rp1 - 2 * En_R + En_Rm1) / (dR**2)

nac01_approx1 = (Cn_R.conj().T @ Cn_Rp1 - np.eye(nbo)) / dR
nac01_approx2 = (np.eye(nbo) - Cn_R.conj().T @ Cn_Rm1) / dR
nac11_approx1 = (2*np.eye(nbo) - Cn_Rp1.conj().T @ Cn_R - Cn_R.conj().T @ Cn_Rp1) / (dR**2)
nac11_approx2 = (2*np.eye(nbo) - Cn_R.conj().T @ Cn_Rm1 - Cn_Rm1.conj().T @ Cn_R) / (dR**2)
nac02_approx0 = (Cn_R.conj().T @ Cn_Rp1 - 2 * np.eye(nbo) + Cn_R.conj().T @ Cn_Rm1) / (dR**2)

n=27
m=27

print(f'R = {RI[I]} a.u.; dR = {dR} a.u.; n = {n}; m = {m}')
print(f'd1En: res: {d1En_R[n]}, 1: {d1En_approx1[n]}, 2: {d1En_approx2[n]}')
print(f'd2En: res: {d2En_R[n]}, 0: {d2En_approx0[n]}')
print(f'nac01: res: {nac01_R[n,m]}, 1: {nac01_approx1[n,m]}, 2: {nac01_approx2[n,m]}')
print(f'nac11: res: {nac11_R[n,m]}, 1: {nac11_approx1[n,m]}, 2: {nac11_approx2[n,m]}')
print(f'nac02: res: {nac02_R[n,m]}, 0: {nac02_approx0[n,m]}')

exit()
kmax = 49
print((nac01_R[n,kmax] * nac01_R[kmax,m]))
print((nac01_R[:,:kmax] @ nac01_R[:kmax,:])[n,m])
print((nac01_R @ nac01_R)[n,m])
