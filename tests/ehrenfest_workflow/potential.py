import numpy as np
from models.two_electron_screened_diatomic import TESD, generate_Hele, generate_dHele, generate_dipole, generate_cap, Vnuc

params = {
    "aee": 0.02,
    "bee": 0.25,
    "aR": 0.01205,
    "bR": 0.01,
    "aAe": 0.0102,
    "bAe": 0.655,
    "aBe": 0.0139,
    "bBe": 0.473,
    "mA": 36443.98900696,
    "mB": 7294.29954142,
    "xmax": 50.0,
    "xpts": 450,
    "xcap": 17.5,
    "etacap": 1.0,
    "ncap": 3,
    "spin": "triplet"
}

model = TESD(params)

Rpts=24
nsta=13
#R=np.linspace(0.4, 15.0, Rpts)
R=np.linspace(0.4, 8.0, Rpts)
ER=np.zeros((Rpts,nsta))
Gamma=np.zeros((Rpts,nsta))
for i in range(Rpts):
    print(i)
    Ri = R[i]
    hcore = model.hcore(Ri)
    Ei, Ci = np.linalg.eigh(hcore)
    ER[i,:]=Ei[:nsta] + model.Vnuc(Ri)
    #ER[i,:]=Ei[:nsta].real
    #Gamma[i,:]=Ei[:nsta].imag

Esinglet=np.zeros((Rpts, 6))
Esinglet[:,0] = ER[:,0] + ER[:,0] - model.Vnuc(R)
Esinglet[:,1] = ER[:,0] + ER[:,1] - model.Vnuc(R)
Esinglet[:,2] = ER[:,1] + ER[:,1] - model.Vnuc(R)
Esinglet[:,3] = ER[:,0] + ER[:,2] - model.Vnuc(R)
Esinglet[:,4] = ER[:,1] + ER[:,2] - model.Vnuc(R)
Esinglet[:,5] = ER[:,2] + ER[:,2] - model.Vnuc(R)
#np.savetxt('ER.dat', Esinglet)

Etriplet=np.zeros((Rpts, 6))
Etriplet[:,0] = ER[:,0] + ER[:,1] - model.Vnuc(R)
Etriplet[:,1] = ER[:,0] + ER[:,2] - model.Vnuc(R)
Etriplet[:,2] = ER[:,1] + ER[:,2] - model.Vnuc(R)
Etriplet[:,3] = ER[:,0] + ER[:,3] - model.Vnuc(R)
Etriplet[:,4] = ER[:,1] + ER[:,3] - model.Vnuc(R)
Etriplet[:,5] = ER[:,2] + ER[:,3] - model.Vnuc(R)
np.savetxt('ER.dat', Etriplet)

np.savetxt('R.dat', R)
#np.savetxt('ER.dat', ER)
np.savetxt('Gamma.dat', Gamma)



#Rpts=7
#nsta=12
#R=np.linspace(1.0, 4.0, Rpts)
##E=np.zeros((Rpts,nsta), dtype=np.complex128)
#ER=np.zeros((Rpts,nsta))
#Gamma=np.zeros((Rpts,nsta))
#for i in range(Rpts):
#    print(i)
#    Ri = R[i]
#    Hele = generate_Hele(Ri, params)
#    #Ei, Ci = np.linalg.eig(Hele)
#    Ei, Ci = np.linalg.eigh(Hele)
#    ER[i,:]=Ei[:nsta] + Vnuc(Ri, params)
#    #ER[i,:]=Ei[:nsta].real
#    #Gamma[i,:]=Ei[:nsta].imag
#
#np.savetxt('R.dat', R)
#np.savetxt('ER.dat', ER)
#np.savetxt('Gamma.dat', Gamma)
