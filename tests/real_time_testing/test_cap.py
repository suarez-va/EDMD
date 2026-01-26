import numpy as np
from scipy.linalg import eig
from models.model_utils import matmat, solve_cap
from models.two_electron_diatomic import TEGD

params = {
    "aR": 0.0,
    "bR": 0.0001,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.0,
    "bee": 0.0001,
}

eta_sub = 1.0e-04
R_sub = 8.0

#model = TEGD(a=-196.7/2.0, b=196.7/2.0, N=500, bounds="(-inf,inf)", spin="singlet", model_params=params)
model = TEGD(a=-196.7/2.0, b=196.7/2.0, N=400, bounds="(-inf,inf)", spin="singlet", model_params=params)

h = model.hij(R = R_sub)
w = model.wij(acap=-50.0, bcap=50.0, ncap=2)

heta = h - 1j * eta_sub * w
hetadag = h + 1j * eta_sub * w

ea, xial, xiar = solve_cap(h, w, eta_sub)
xia = xiar
xai_inv = xial.conj().T
ea_dag, xial, xiar = solve_cap(h, w, -eta_sub)
xia_invdag = xiar
xai_dag = xial.conj().T

print(np.max(np.abs(ea.conj()-ea_dag)))

etestdag = xia.conj().T @ hetadag @ xai_inv.conj().T
htestdag=np.einsum('ia,a,aj->ij', xai_inv.conj().T, ea_dag, xia.conj().T)
print(np.max(np.abs(etestdag-np.diag(ea_dag))))
print(np.max(np.abs(htestdag-hetadag)))

#print(np.max(np.abs(xia.conj().T-xai_dag)))
exit()

htest=np.einsum('ia,a,aj->ij', xia, ea, xai_inv)
htestdag=np.einsum('ia,a,aj->ij', xia_invdag, ea_dag, xai_dag)
print(np.max(np.abs(htest-heta)))
print(np.max(np.abs(htestdag-hetadag)))

exit()


xiartest = np.linalg.inv(xial.conj().T)
print(np.max(np.abs(xiartest - xiar)))
Sab = xial.conj().T @ xiar 
Sij = xiar @ xial.conj().T 
print(np.max(np.abs(Sab - np.eye(ea.shape[0]))))
print(np.max(np.abs(Sij - np.eye(ea.shape[0]))))
#ovlp = xiar @ xial.conj().T
etest = xial.conj().T @ heta @ xiar
print(np.max(np.abs(etest-np.diag(ea))))
htest=np.einsum('ia,a,aj->ij', xiar, ea, xial.conj().T)
print(np.max(np.abs(htest-heta)))

exit()

ea, xial, xajr = eig(heta, left=True, right=True)
norm = np.sqrt(np.diag(np.matmul(xial.conj().T, xajr)))
#xial *= 1 / norm; xajr *= 1 / norm.conj()
xial *= 1 / norm.conj(); xajr *= 1 / norm
etest = np.matmul(xial.conj().T, np.matmul(heta, xajr))
print(np.max(np.abs(etest-np.diag(ea))))
exit()



ea, xial, xajr = eig(heta, left=True, right=True)
S = np.matmul(xial.conj().T, xajr)
for j in range(len(ea)):
    xial[:, j] /= S[j, j].conj()
etest = np.matmul(xial.conj().T, np.matmul(heta, xajr))
print(np.max(np.abs(etest-np.diag(ea))))
exit()




#etest = np.matmul(xial.conj().T, np.matmul(heta, xajr))
#etest = np.matmul(xajr.conj().T, np.matmul(heta, xial))
#etest = np.matmul(xajr, np.matmul(heta, xial.conj().T))
#etest, xjunk = np.linalg.eig(heta)
#idx = np.argsort(etest.real)
#etest = etest[idx]

#print(np.max(np.absolute(etest-np.diag(ea))))


#print(np.matmul(xial.conj().T, xajr))

#htest=np.einsum('ia,a,aj->ij', xial.conj().T, ea, xajr)
#htest=np.einsum('ia,a,aj->ij', xial, ea, xajr.conj().T)
#htest=np.einsum('ia,a,aj->ij', xajr, ea, xial.conj().T)

print(np.max(np.absolute(htest-heta)))
exit()

#ep, cip = model.solve_mos(R = 8.0)

#exit()

CIn = model.solve_wfn(R = 8.0, nbo = 250)
Wab50n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-50.0, bcap=50.0, ncap=2), CIn))
np.savez("boops", Wab50n2=Wab50n2)

exit()

ep, cip = model.solve_mos(R = 8.0)

