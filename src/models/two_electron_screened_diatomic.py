import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../src"))
from colbert_miller_dvr import dvr_xn, dvr_T, dvr_W


def map_CI(xpts):
    # generate index mapping between single particle grid points and DVR slater determinants
    mapping = np.full((xpts, xpts), -1)
    k = 0
    for i in range(xpts):
        for j in range(i+1,xpts):
            mapping[i,j] = k
            k += 1

    return mapping


def generate_Hele(R, params):

    aee = params["aee"]
    bee = params["bee"]
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

    # generate hcore using Colbert-Miller syle DVR for kinetic energy
    hcore = dvr_T(mu, -xmax, xmax, xpts-1, "(-inf,inf)")
    for i in range(xpts):
        xi = -xmax + dx * i; Aarg = (xi + mu / mA * R); Barg = (xi - mu / mB * R)
        hcore[i,i] += -np.exp(-aAe * Aarg**2) / np.sqrt(Aarg**2 + bAe)
        hcore[i,i] += -np.exp(-aBe * Barg**2) / np.sqrt(Barg**2 + bBe)

    # generate full CI Hele matrix
    nstates = int(xpts * (xpts - 1) / 2)
    mapping = map_CI(xpts)
    Hele = np.zeros((nstates, nstates), dtype=complex)
    # iket == ibra and jket == jbra
    for i in range(xpts):
        for j in range(i+1,xpts):
            xi = -xmax + dx * i; xj = -xmax + dx * j; xarg = (xi - xj)**2
            Hele[mapping[i,j],mapping[i,j]] += hcore[i,i] + hcore[j,j]
            Hele[mapping[i,j],mapping[i,j]] += np.exp(-aee * xarg) / np.sqrt(xarg + bee)
    # iket == ibra but jket != jbra
    for i in range(xpts):
        for jket in range(i+1,xpts):
            for jbra in range(i+1,xpts):
                if jket == jbra:
                    pass
                else:
                    Hele[mapping[i,jket],mapping[i,jbra]] += hcore[jket,jbra]
    # iket != ibra but jket == jbra
    for iket in range(xpts):
        for j in range(iket+1,xpts):
            for ibra in range(j):
                if iket == ibra:
                    pass
                else:
                    Hele[mapping[iket,j],mapping[ibra,j]] += hcore[iket,ibra]

    return Hele


def generate_dHele(R, params):

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

    # generate dhcore
    hcore = np.zeros((xpts, xpts), dtype=complex)
    for i in range(xpts):
        xi = -xmax + dx * i; Aarg = (xi + mu / mA * R); Barg = (xi - mu / mB * R)
        hcore[i,i] += mu / mA * (2 * aAe + 1 / (Aarg**2 + bAe)) * Aarg * np.exp(-aAe * Aarg**2) / np.sqrt(Aarg**2 + bAe)
        hcore[i,i] += -mu / mB * (2 * aBe + 1 / (Barg**2 + bBe)) * Barg * np.exp(-aBe * Barg**2) / np.sqrt(Barg**2 + bBe)

    # generate full CI dHele matrix
    nstates = int(xpts * (xpts - 1) / 2)
    mapping = map_CI(xpts)
    Hele = np.zeros((nstates, nstates), dtype=complex)
    # iket == ibra and jket == jbra
    for i in range(xpts):
        for j in range(i+1,xpts):
            Hele[mapping[i,j],mapping[i,j]] += hcore[i,i] + hcore[j,j]

    return Hele


def calculate_nac(E, dHele_adi):
    nstates = E.shape[0]
    nac = np.zeros((nstates, nstates), dtype=complex)
    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                pass
            else:
                nac[i,j] = dHele_adi[i,j] / (E[j] - E[i])
    return nac


def generate_dipole(R, params):

    xmax = params["xmax"]
    xpts = params["xpts"]

    dx = (2 * xmax) / (xpts - 1)

    # generate 1 electron transition dipole matrix in the Colbert-Miller DVR basis
    hcore = -dvr_xn(1, -xmax, xmax, xpts-1, "(-inf,inf)")

    # generate full CI dipole matrix
    nstates = int(xpts * (xpts - 1) / 2)
    mapping = map_CI(xpts)
    dipole = np.zeros((nstates, nstates), dtype=complex)
    for i in range(xpts):
        for j in range(i+1,xpts):
            dipole[mapping[i,j],mapping[i,j]] += hcore[i,i] + hcore[j,j]
            
    return dipole


def generate_cap(R, params):

    xmax = params["xmax"]
    xpts = params["xpts"]
    xcap = params["xcap"]
    etacap = params["etacap"]
    ncap = params["ncap"]

    dx = (2 * xmax) / (xpts - 1)

    # generate 1 electron CAP matrix in the Colbert-Miller DVR basis
    hcore = dvr_W(-xmax, xmax, xpts-1, -xcap, xcap, etacap, ncap, "(-inf,inf)")

    # generate full CI CAP matrix
    nstates = int(xpts * (xpts - 1) / 2)
    mapping = map_CI(xpts)
    cap = np.zeros((nstates, nstates), dtype=complex)
    for i in range(xpts):
        for j in range(i+1,xpts):
            cap[mapping[i,j],mapping[i,j]] += hcore[i,i] + hcore[j,j]
            
    return cap

