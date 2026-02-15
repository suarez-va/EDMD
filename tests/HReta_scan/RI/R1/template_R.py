from model_systems.models import GICD
from time_independent.one_electron_fixed_nuclei import OneElectronFixedNuclei

params = {
    'ZA': 0.5,
    'ZB': 0.5,
    'aR': 0.0,
    'bR': 0.0001,
    'DA' : 1.0,
    'bA': 0.25,
    'DB' : 0.8,
    'bB': 1.0,
    'aee': 0.0,
    'bee': 0.0001,
}

model = GICD(params)
oefn = OneElectronFixedNuclei(model=model, xa=-196.7/2.0, xb=196.7/2.0, xN=500, xbounds="(-inf,inf)")
oefn.solve_mos(R = 3.01000000000, nmo = oefn.nxdvr)


#Cn = model.solve_bo(R = 3.01000000000, nbo = 500)

#Wab50n2 = np.matmul(Cn.conj().T, matmat(model.W(acap=-50.0, bcap=50.0, ncap=2), Cn))
#np.savez("boops", Wab50n2=Wab50n2)

#Wab20n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-20.0, bcap=20.0, ncap=2), CIn))
#Wab40n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-40.0, bcap=40.0, ncap=2), CIn))
#Wab50n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-50.0, bcap=50.0, ncap=2), CIn))
#Wab60n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-60.0, bcap=60.0, ncap=2), CIn))
#Wab80n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-80.0, bcap=80.0, ncap=2), CIn))
#Wab100n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-100.0, bcap=100.0, ncap=2), CIn))
#Wab120n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-120.0, bcap=120.0, ncap=2), CIn))
#Wab150n2 = np.matmul(CIn.conj().T, matmat(model.W(acap=-150.0, bcap=150.0, ncap=2), CIn))
#
#np.savez("boops", Wab20n2=Wab20n2, Wab40n2=Wab40n2, Wab50n2=Wab50n2, Wab60n2=Wab60n2, Wab80n2=Wab80n2, Wab100n2=Wab100n2, Wab120n2=Wab120n2, Wab150n2=Wab150n2)

