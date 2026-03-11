import numpy as np
from dvr_basis.colbert_miller_dvr import dvr_p, dvr_T
from scipy.linalg import eigh

def new_T(m, a, b, N):

    dx = (b - a) / N

    Tij = np.zeros((N+1, N+1), dtype=np.complex128)
    for iket in range(N+1):
        for ibra in range(N+1):
            if (iket == ibra):
                Tij[iket,ibra] += 2 / (2 * m * dx**2)
            elif (iket == ibra - 1): 
                Tij[iket,ibra] += -1 / (2 * m * dx**2)
            elif (iket == ibra + 1): 
                Tij[iket,ibra] += -1 / (2 * m * dx**2)

    return Tij


Ntest = 50
ni=0
nf=5
nidx = ni + np.arange(int(nf-ni))
Eexact = nidx
m = 0.0
atest = 0.0
btest = 0.0
vab = np.zeros((Ntest-1))
vinf = np.zeros((Ntest+1))

sys = 1

# particle in a box
if (sys == 0):
    m = 2.5
    L = 5.0
    V0 = 10000000000000.0
    Eexact = np.pi**2/(2*m*L**2)*(nidx + 1)**2
    ishift = 1
    atest = -(ishift/(Ntest-2*ishift))*L
    btest = ((Ntest-ishift)/(Ntest-2*ishift))*L
    dx = (btest - atest) / Ntest
    xab = np.linspace(atest+dx,btest-dx,Ntest-1); xab[ishift-1] = 0.0; xab[Ntest-ishift-1] = L
    xinf = np.linspace(atest,btest,Ntest+1); xinf[ishift] = 0.0; xinf[Ntest-ishift] = L
    vab = V0 * (np.heaviside(-xab, 0.5) + np.heaviside(xab - L, 0.5))
    vinf = V0 * (np.heaviside(-xinf, 0.5) + np.heaviside(xinf - L, 0.5))
    #print(vab[ishift-2], vab[ishift-1], vab[ishift], vab[Ntest-ishift-2], vab[Ntest-ishift-1], vab[Ntest-ishift])
    print(vinf[ishift-1], vinf[ishift], vinf[ishift+1], vinf[Ntest-ishift-1], vinf[Ntest-ishift], vinf[Ntest-ishift+1])

# harmonic oscillator
if (sys == 1):
    m = 2.5
    w = 1.25
    Eexact = w*(nidx + 0.5)
    sigma = 1/np.sqrt(2*m*w)
    atest = -10*sigma
    btest = 10*sigma
    dx = (btest - atest) / Ntest
    xab = np.linspace(atest+dx,btest-dx,Ntest-1)
    xinf = np.linspace(atest,btest,Ntest+1)
    vab = 0.5*m*w**2*xab**2
    vinf = 0.5*m*w**2*xinf**2

#p = dvr_p(a=atest, b=btest, N=Ntest, bounds='(-inf,inf)')
Tab = dvr_T(m, a=atest, b=btest, N=Ntest, bounds='(a,b)')
Tinf = dvr_T(m, a=atest, b=btest, N=Ntest, bounds='(-inf,inf)')
Tnew = new_T(m, atest, btest, Ntest)

Hab = Tab + np.diag(vab)
Hinf = Tinf + np.diag(vinf)
Hnew = Tnew + np.diag(vinf)

Eab, Vab = eigh(Hab)
Einf, Vinf = eigh(Hinf)
Enew, Vnew = eigh(Hnew)

print(f'Eab: {Eab[nidx]}')
print(f'Einf: {Einf[nidx]}')
#print(f'Enew: {Enew[nidx]}')
print(f'Eexact: {Eexact}')
print(Vnew[0,0], Vnew[1,0], Vnew[2,0])
exit()


#T1 = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
#T2 = np.array([[np.pi**2/3,-2,1/2],[-2,np.pi**2/3,-2],[1/2,-2,np.pi**2/3]])
#print(T1)
#print(Tnew)
#print(T2)
#print(T)

#P = dvr_p(a=-1.0, b=1.0, N=2, bounds='(-inf,inf)')
#E, V = np.linalg.eigh(P)
#print(P)
#print(E)
#
#T = dvr_T(0.5, a=-1.0, b=1.0, N=2, bounds='(-inf,inf)')
#E, V = np.linalg.eigh(T)
#print(T)
#print(E)



k, v = np.linalg.eigh(p)
k2, V = np.linalg.eigh(T)
k2new, Vnew = np.linalg.eigh(Tnew)

k2test = k**2
idx = np.argsort(k2test.real)
k2test = k2test[idx]


ni=0
nf=10
#print(k[ni:nf])
#print(k2test[ni:nf])
print(k2[ni:nf])
print(k2new[ni:nf])

exit()

T1 = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
T2 = np.array([[np.pi**2/3,-2,1/2],[-2,np.pi**2/3,-2],[1/2,-2,np.pi**2/3]])
print(T1)
print(T2)

e, v = np.linalg.eigh(T1)
print(e-T1[0,0])

e, v = np.linalg.eigh(T2)
print(e-T2[0,0])
