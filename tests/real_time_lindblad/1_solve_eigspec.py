import numpy as np
from model_systems.models import GICD
from time_independent.fci.one_electron_fixed_nuclei import OneElectronFixedNuclei
from time_independent.fci.two_electron_fixed_nuclei import TwoElectronFixedNuclei

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

R = 8.0
acap = -50.0
bcap = 50.0
ncap = 2

model = GICD(params)
oefn = OneElectronFixedNuclei(model=model, xa=-196.7/2.0, xb=196.7/2.0, xN=50, xbounds='(-inf,inf)')
tefn = TwoElectronFixedNuclei(model=model, xa=-196.7/2.0, xb=196.7/2.0, xN=50, xbounds='(-inf,inf)', spin='singlet')
oefn.solve_fci(R = R, nbo = 5)
tefn.solve_fci(R = R, nbo = 22)


#cip = model.solve_mos(R = R)
#wij = model.wij(acap = acap, bcap = bcap, ncap = ncap)
#wk = np.diag(wij)
#wpq = np.matmul(cip.conj().T, np.matmul(wij, cip))
#np.savez("moops", wk = wk, wpq = wpq)

#Cn = model.solve_bo(R = R, nbo = 225)
#Wnm = np.matmul(Cn.conj().T, matmat(model.W(acap = acap, bcap = bcap, ncap = ncap), Cn))
#np.savez("boops", Wnm=Wnm)

