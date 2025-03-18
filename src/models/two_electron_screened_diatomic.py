import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))

from colbert_miller_dvr import dvr_T


def generate_Hele(R, params):

    aee = params["aee"]
    bee = params["bee"]
    aR = params["aR"]
    bR = params["bR"]
    aAe = params["aAe"]
    bAe = params["bAe"]
    aBe = params["aBe"]
    bBe = params["bBe"]
    mA = params["mA"]
    mB = params["mB"]
    xmax = params["xmax"]
    xpts = params["xpts"]

    mu = mA * mB / (mA + mB)
    dx = (2 * xmax) / (xpts - 1)
    nstates = int(xpts * (xpts - 1) / 2)

    # First generate hcore using Colbert-Miller syle DVR for kinetic energy
    hcore = dvr_T(mu, -xmax, xmax, xpts, limit="(-inf,inf)")
    for i in range(xpts):
        xi = -xmax + dx * i; Aarg = (xi + mu / mA * R)**2; Barg = (xi - mu / mB * R)**2
        hcore[i,i] += -np.exp(-aAe * Aarg) / np.sqrt(Aarg + bAe)
        hcore[i,i] += -np.exp(-aBe * Barg) / np.sqrt(Barg + bBe)

    # Next generate index Map between single particle grid points and DVR slater determinants
    Map = np.full((xpts, xpts), -1)
    k = 0
    for i in range(xpts):
        for j in range(i+1,xpts):
            Map[i,j] = k
            k += 1

    # Finally, generate full CI Hamiltonian
    Hele = np.zeros((nstates, nstates), dtype=complex)
    # iket == ibra and jket == jbra
    for i in range(xpts):
        for j in range(i+1,xpts):
            xi = -xmax + dx * i; xj = -xmax + dx * j; xarg = (xi - xj)**2
            Hele[Map[i,j],Map[i,j]] += hcore[i,i] + hcore[j,j]
            Hele[Map[i,j],Map[i,j]] += np.exp(-aee * xarg) / np.sqrt(xarg + bee)
    # iket == ibra but jket != jbra
    for i in range(xpts):
        for jket in range(i+1,xpts):
            for jbra in range(i+1,xpts):
                if jket == jbra:
                    pass
                else:
                    Hele[Map[i,jket],Map[i,jbra]] += hcore[jket,jbra]
    # iket != ibra but jket == jbra
    for iket in range(xpts):
        for j in range(iket+1,xpts):
            for ibra in range(j):
                if iket == ibra:
                    pass
                else:
                    Hele[Map[iket,j],Map[ibra,j]] += hcore[iket,ibra]

    return Hele


