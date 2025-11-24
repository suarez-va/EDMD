import sys
import os
import numpy as np

#sys.path.append(os.path.abspath("../src"))
from grid_utils.colbert_miller_dvr import dvr_xn, dvr_T, dvr_W
from models.model_utils import Model

class TESD(Model):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.ndvr = self.params["ndvr"]
        self.a = -self.params["xmax"]
        self.b = self.params["xmax"]
        self.N = self.ndvr - 1
        self.bounds = "(-inf,inf)"
        self.ep = np.zeros((self.ndvr),dtype=np.complex128)
        self.cip = np.zeros((self.ndvr,self.ndvr),dtype=np.complex128)

        self.spin = self.params["spin"]
        assert self.spin in ("singlet", "triplet"), f"Spin multiplicity: {self.spin}, must be 'singlet' or 'triplet'."
        self.nfci = int((self.ndvr+1)*self.ndvr/2) if self.spin=="singlet" else int(self.ndvr*(self.ndvr-1)/2) if self.spin=="triplet" else 0
        self.map_ij, self.map_kl = np.triu_indices(self.ndvr, k=0) if self.spin=="singlet" else np.triu_indices(self.ndvr, k=1) if self.spin=="triplet" else np.zeros((self.nfci), dtype=int)
        self.map_ikjl = np.full((self.ndvr, self.ndvr), -1, dtype=int)
        self.map_ikjl[self.map_ij, self.map_kl] = np.arange(self.nfci)
        self.En = np.zeros((self.nfci),dtype=np.complex128)
        self.Cijn = np.zeros((self.ndvr,self.ndvr,self.nfci),dtype=np.complex128)

        self.eta = self.params["eta"]
        self.acap = -self.params["xcap"]
        self.bcap = self.params["xcap"]
        self.ncap = self.params["ncap"]

        #self.ncas = self.params["ncas"]

    def VR(self, R):
        aR = self.params["aR"]
        bR = self.params["bR"]
        return np.exp(-aR * R**2) / np.sqrt(R**2 + bR)

    def xi(self):
        return np.linspace(self.a, self.b, self.ndvr)

    def VeR(self, x, R):
        aAe = self.params["aAe"]
        bAe = self.params["bAe"]
        aBe = self.params["aBe"]
        bBe = self.params["bBe"]
        mA = self.params["mA"]
        mB = self.params["mB"]
        mu = mA * mB / (mA + mB)
        Aarg = (x + mu / mA * R)**2; Barg = (x - mu / mB * R)**2
        return -np.exp(-aAe * Aarg) / np.sqrt(Aarg + bAe) - np.exp(-aBe * Barg) / np.sqrt(Barg + bBe)

    def hij(self, R):
        # generate hcore using Colbert-Miller syle DVR for kinetic energy
        h = dvr_T(1, self.a, self.b, self.N, self.bounds) + np.diag(self.VeR(self.xi(), R))
        if self.eta != 0.0:
            h += dvr_W(self.a, self.b, self.N, self.acap, self.bcap, self.eta, self.ncap, self.bounds)
        return h

    def solve_mos(self, R):
        if self.eta == 0.0:
            self.ep[:], self.cip = np.linalg.eigh(self.hij(R))
        else:
            e, c = np.linalg.eig(self.hij(R))
            idx = np.argsort(e.real)
            self.ep, self.cip = e[idx], c[:,idx]

    def Vee(self, x1, x2):
        aee = self.params["aee"]
        bee = self.params["bee"]
        xarg = (x1 - x2)**2
        return np.exp(-aee * xarg) / np.sqrt(xarg + bee)
        #return 0 * np.exp(-aee * xarg) / np.sqrt(xarg + bee)

    def Vik(self):
        # generate hcore using Colbert-Miller syle DVR for kinetic energy
        xi=self.xi()
        return self.Vee(xi[:,None],xi[None,:])

    def Hikjl(self, R):
        Hikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        hij = self.hij(R)
        Vik = self.Vik()
        Hikjl += hij[:,None,:,None] * dij[None,:,None,:]
        Hikjl += dij[:,None,:,None] * hij[None,:,None,:]
        Hikjl += Vik[:,:,None,None] * dij[:,None,:,None] * dij[None,:,None,:]
        return Hikjl

    def Hdvr(self, R):
        H = np.zeros((self.nfci, self.nfci), dtype=np.complex128)
        dij = np.eye(self.ndvr)
        Hikjl = self.Hikjl(R)
        match self.spin:
            case "singlet":
                Hikjl *= (1 - (1 - 1/np.sqrt(2)) * dij[:,:,None,None]) * (1 - (1 - 1/np.sqrt(2)) * dij[None,None,:,:])
                H = Hikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_ij[None,:],self.map_kl[None,:]] + Hikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_kl[None,:],self.map_ij[None,:]]
            case "triplet":
                H = Hikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_ij[None,:],self.map_kl[None,:]] - Hikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_kl[None,:],self.map_ij[None,:]]
        return H

    def solve_wfn(self, R):
        if self.eta == 0.0:
            self.En[:], self.Cijn[self.map_ij, self.map_kl, :] = np.linalg.eigh(self.Hdvr(R))
        else:
            E, C = np.linalg.eig(self.Hdvr(R))
            idx = np.argsort(E.real)
            self.En, self.Cijn[self.map_ij, self.map_kl, :] = E[idx], C[:,idx]
        match self.spin:
            case "singlet":
                self.Cijn += self.Cijn.swapaxes(0,1)
                self.Cijn *= 1 / np.sqrt(2)
                self.Cijn[np.arange(self.ndvr),np.arange(self.ndvr),:] *= 1 / np.sqrt(2)
            case "triplet":
                self.Cijn += -self.Cijn.swapaxes(0,1)
                self.Cijn *= 1 / np.sqrt(2)

#    def wij(self):
#        # generate complex absorbing potential using Colbert-Miller syle DVR
#        return dvr_W(self.a, self.b, self.N, self.acap, self.bcap, self.eta, self.ncap, self.bounds)
#
#    def Wikjl(self):
#        Wikjl = np.zeros((self.ndvr, self.ndvr, self.ndvr, self.ndvr), dtype=np.complex128)
#        dij = np.eye(self.ndvr)
#        wij = self.wij()
#        Wikjl += wij[:,None,:,None] * dij[None,:,None,:]
#        Wikjl += dij[:,None,:,None] * wij[None,:,None,:]
#        return Wikjl
#
#    def Wdvr(self):
#        W = np.zeros((self.nfci, self.nfci), dtype=np.complex128)
#        dij = np.eye(self.ndvr)
#        Wikjl = self.Wikjl()
#        match self.spin:
#            case "singlet":
#                Wikjl *= (1 - (1 - 1/np.sqrt(2)) * dij[:,:,None,None]) * (1 - (1 - 1/np.sqrt(2)) * dij[None,None,:,:])
#                W = Wikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_ij[None,:],self.map_kl[None,:]] + Wikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_kl[None,:],self.map_ij[None,:]]
#            case "triplet":
#                W = Wikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_ij[None,:],self.map_kl[None,:]] - Wikjl[self.map_ij[:,None],self.map_kl[:,None],self.map_kl[None,:],self.map_ij[None,:]]
#        return W



#    def Hdvr(self, R):
#        spin = self.params["spin"]
#        nmpts = int((self.ndvr+1)*self.ndvr/2) if spin=="singlet" else int(self.ndvr*(self.ndvr-1)/2) if spin=="triplet" else 0
#        H = np.zeros((nmpts, nmpts), dtype=np.complex128)
#        dij = np.eye(self.ndvr)
#        Hikjl = self.Hikjl(R)
#        map_ij = np.zeros(nmpts, dtype=int)
#        map_kl = np.zeros(nmpts, dtype=int)
#        nm = 0
#        match spin:
#            case "singlet":
#                for ij in range(self.ndvr):
#                    for kl in range(ij,self.ndvr):
#                        map_ij[nm] = ij
#                        map_kl[nm] = kl
#                        nm += 1
#                Hikjl *= (1 - (1 - 1/np.sqrt(2)) * dij[:,:,None,None]) * (1 - (1 - 1/np.sqrt(2)) * dij[None,None,:,:])
#                H = Hikjl[map_ij[:,None],map_kl[:,None],map_ij[None,:],map_kl[None,:]] + Hikjl[map_ij[:,None],map_kl[:,None],map_kl[None,:],map_ij[None,:]]
#            case "triplet":
#                for ij in range(self.ndvr):
#                    for kl in range(ij+1,self.ndvr):
#                        map_ij[nm] = ij
#                        map_kl[nm] = kl
#                        nm += 1
#                H = Hikjl[map_ij[:,None],map_kl[:,None],map_ij[None,:],map_kl[None,:]] - Hikjl[map_ij[:,None],map_kl[:,None],map_kl[None,:],map_ij[None,:]]
#        return H

    def Hprqs(self, R):
        Hprqs = np.zeros((self.ncas, self.ncas, self.ncas, self.ncas),dtype=np.complex128)
        self.solve_mos(R)
        dpq = np.eye(self.ncas)
        Hprqs += (self.ep[:self.ncas,None,None,None] + self.ep[None,:self.ncas,None,None]) * dpq[:,None,:,None] * dpq[None,:,None,:]
        xi=self.xi()
        Vik = self.Vee(xi[:,None],xi[None,:])
        CiN = np.einsum('ip,iq->ipq',self.cip[:,:self.ncas].conj(),self.cip[:,:self.ncas]).reshape(self.ndvr, self.ncas**2)
        VNM = np.linalg.multi_dot([CiN.T, Vik, CiN])
        Hprqs += VNM.reshape(self.ncas,self.ncas,self.ncas,self.ncas).swapaxes(1,2)
        return Hprqs

    def map_casci(self):
        # generate index mapping between single particle grid points and DVR slater determinants
        spin = self.params["spin"]
        nmpts = int((self.ncas+1)*self.ncas/2) if spin=="singlet" else int(self.ncas*(self.ncas-1)/2) if spin=="triplet" else 0
        map_pq = np.zeros(nmpts, dtype=int)
        map_rs = np.zeros(nmpts, dtype=int)
        map_prqs = np.full((self.ncas, self.ncas), -1)
        nm = 0
        match spin:
            case "singlet":
                for pq in range(self.ncas):
                    for rs in range(pq,self.ncas):
                        map_pq[nm] = pq
                        map_rs[nm] = rs
                        map_prqs[pq,rs] = nm
                        nm += 1
            case "triplet":
                for pq in range(self.ncas):
                    for rs in range(pq+1,self.ncas):
                        map_pq[nm] = pq
                        map_rs[nm] = rs
                        map_prqs[pq,rs] = nm
                        nm += 1

        return map_pq, map_rs, map_prqs

    # THIS ON STILL NEEDS TO BE FIXED FOR SINGLETS
    def Hele(self, R):
        spin = self.params["spin"]
        map_pq, map_rs, map_prqs = self.map_casci()
        self.solve_mos(R)
        Hprqs = self.Hprqs()
        match spin:
            case "singlet":
                nstates = int((self.ncas + 1) * self.ncas / 2)
                H = np.zeros((nstates, nstates), dtype=np.complex128)
                dpq = np.eye(self.ncas)
                # THIS IS WRONG, ONLY WORKS FOR DIAGONAL HCORE!!!
                Hprqs *= 1 - 0.5 * dpq[:,None,:,None] * dpq[None,:,None,:] * dpq[:,None,None,:] * dpq[None,:,:,None]
                H = Hprqs[map_pq[:,None],map_rs[:,None],map_pq[None,:],map_rs[None,:]] + Hprqs[map_pq[:,None],map_rs[:,None],map_rs[None,:],map_pq[None,:]]
            case "triplet":
                nstates = int(self.ncas * (self.ncas - 1) / 2)
                H = np.zeros((nstates, nstates), dtype=np.complex128)
                dpq = np.eye(self.ncas)
                H = Hprqs[map_pq[:,None],map_rs[:,None],map_pq[None,:],map_rs[None,:]] - Hprqs[map_pq[:,None],map_rs[:,None],map_rs[None,:],map_pq[None,:]]
        return H

#    def ci_map(self):
#        # generate index mapping between single particle grid points and DVR slater determinants
#        map_ij = np.zeros(self.nfci, dtype=int)
#        map_kl = np.zeros(self.nfci, dtype=int)
#        map_ikjl = np.full((self.ndvr, self.ndvr), -1)
#        nm = 0
#        match self.spin:
#            case "singlet":
#                for ij in range(self.ndvr):
#                    for kl in range(ij,self.ndvr):
#                        map_ij[nm] = ij
#                        map_kl[nm] = kl
#                        map_ikjl[ij,kl] = nm
#                        nm += 1
#            case "triplet":
#                for ij in range(self.ndvr):
#                    for kl in range(ij+1,self.ndvr):
#                        map_ij[nm] = ij
#                        map_kl[nm] = kl
#                        map_ikjl[ij,kl] = nm
#                        nm += 1
#        return map_ij, map_kl, map_ikjl




#    def hcore(self, R):
#        aAe = self.params["aAe"]
#        bAe = self.params["bAe"]
#        aBe = self.params["aBe"]
#        bBe = self.params["bBe"]
#        mA = self.params["mA"]
#        mB = self.params["mB"]
#        xmax = self.params["xmax"]
#        xpts = self.params["xpts"]
#    
#        mu = mA * mB / (mA + mB)
#        dx = (2 * xmax) / (xpts - 1)
#    
#        # generate hcore using Colbert-Miller syle DVR for kinetic energy
#        h = dvr_T(1, -xmax, xmax, xpts-1, "(-inf,inf)")
#        for i in range(xpts):
#            xi = -xmax + dx * i; Aarg = (xi + mu / mA * R); Barg = (xi - mu / mB * R)
#            h[i,i] += -np.exp(-aAe * Aarg**2) / np.sqrt(Aarg**2 + bAe)
#            h[i,i] += -np.exp(-aBe * Barg**2) / np.sqrt(Barg**2 + bBe)
#        return h

#    def Hprqs2(self):
#        Hprqs = np.zeros((self.ncas, self.ncas, self.ncas, self.ncas),dtype=np.complex128)
#        dpq = np.eye(self.ncas)
#        Hprqs += (self.ep[:self.ncas,None,None,None] + self.ep[None,:self.ncas,None,None]) * dpq[:,None,:,None] * dpq[None,:,None,:]
#        xi=self.xi()
#        Vik = self.Vee(xi[:,None],xi[None,:])
#        Cipq = np.einsum('ip,iq->ipq',self.cip[:,:self.ncas].conj(),self.cip[:,:self.ncas])
#        Virs = np.einsum('ik,krs->irs',Vik, Cipq)
#        Hprqs += np.einsum('ipq,irs->prqs', Cipq, Virs)
#        return Hprqs


#    def map_ci_dvr(self):
#        # generate index mapping between single particle grid points and DVR slater determinants
#        xpts = self.params["xpts"]
#        spin = self.params["spin"]
#        map = np.full((xpts, xpts), -1)
#        k = 0
#        match spin:
#            case "singlet":
#                for i in range(xpts):
#                    for j in range(i,xpts):
#                        map[i,j] = k
#                        k += 1
#            case "triplet":
#                for i in range(xpts):
#                    for j in range(i+1,xpts):
#                        map[i,j] = k
#                        k += 1
#        return map


#    def Hele(self, R):
#        aee = self.params["aee"]
#        bee = self.params["bee"]
#        xmax = self.params["xmax"]
#        xpts = self.params["xpts"]
#        spin = self.params["spin"]
#    
#        dx = (2 * xmax) / (xpts - 1)
#        # generate hcore using Colbert-Miller syle DVR for kinetic energy
#        h = self.hcore(R) 
##        # generate full CI Hele matrix
#        map = self.map_ci()
#        match spin:
#            case "singlet":
#                nstates = int((xpts + 1) * xpts / 2)
#                H = np.zeros((nstates, nstates), dtype=complex)
#                for iket in range(xpts):
#                    for jket in range(iket,xpts):
#                        for ibra in range(xpts):
#                            for jbra in range(ibra,xpts):
#                                xi = -xmax + dx * iket; xj = -xmax + dx * jket; xarg = (xi - xj)**2
#                                dii = 1 if iket == ibra else 0
#                                djj = 1 if jket == jbra else 0
#                                dij = 1 if iket == jbra else 0
#                                dji = 1 if jket == ibra else 0
#                                H[map[iket,jket],map[ibra,jbra]] += dii*h[jket,jbra] + djj*h[iket,ibra] + dij*h[jket,ibra] + dji*h[iket,jbra]
#                                #H[map[iket,jket],map[ibra,jbra]] += (dii*djj+dij*dji)*np.exp(-aee*xarg)/np.sqrt(xarg+bee)
#                                if dii == 1 and djj == 1 and dij == 1:
#                                    H[map[iket,jket],map[ibra,jbra]] *= 0.5
#            case "triplet":
#                nstates = int(xpts * (xpts - 1) / 2)
#                H = np.zeros((nstates, nstates), dtype=complex)
#                for iket in range(xpts):
#                    for jket in range(iket+1,xpts):
#                        for ibra in range(xpts):
#                            for jbra in range(ibra+1,xpts):
#                                xi = -xmax + dx * iket; xj = -xmax + dx * jket; xarg = (xi - xj)**2
#                                dii = 1 if iket == ibra else 0
#                                djj = 1 if jket == jbra else 0
#                                dij = 1 if iket == jbra else 0
#                                dji = 1 if jket == ibra else 0
#                                H[map[iket,jket],map[ibra,jbra]] = dii*h[jket,jbra] + djj*h[iket,ibra] - dij*h[jket,ibra] - dji*h[iket,jbra]
#                                #H[map[iket,jket],map[ibra,jbra]] += (dii*djj-dij*dji)*np.exp(-aee*xarg)/np.sqrt(xarg+bee)
#        return H

    def ovlp(self, R):
        xmax = self.params["xmax"]
        xpts = self.params["xpts"]
        spin = self.params["spin"]
    
        dx = (2 * xmax) / (xpts - 1)
        # generate full CI Hele matrix
        map = self.map_ci()
        print(map)
        match spin:
            case "singlet":
                nstates = int((xpts + 1) * xpts / 2)
                H = np.zeros((nstates, nstates))
                for iket in range(xpts):
                    for jket in range(iket,xpts):
                        for ibra in range(xpts):
                            for jbra in range(ibra,xpts):
                                #print((map[iket,jket],map[ibra,jbra]))
                                dii = 1 if iket == ibra else 0
                                djj = 1 if jket == jbra else 0
                                dij = 1 if iket == jbra else 0
                                dji = 1 if jket == ibra else 0
                                H[map[iket,jket],map[ibra,jbra]] = dii*djj + dij*dji
                                #H[map[iket,jket],map[ibra,jbra]] += (dii*djj+dij*dji)*np.exp(-aee*xarg)/np.sqrt(xarg+bee)
                                if dii == 1 and djj == 1 and dij == 1:
                                    H[map[iket,jket],map[ibra,jbra]] *= 0.5
            case "triplet":
                nstates = int(xpts * (xpts - 1) / 2)
                H = np.zeros((nstates, nstates), dtype=complex)
                for iket in range(xpts):
                    for jket in range(iket+1,xpts):
                        for ibra in range(xpts):
                            for jbra in range(ibra+1,xpts):
                                dii = 1 if iket == ibra else 0
                                djj = 1 if jket == jbra else 0
                                dij = 1 if iket == jbra else 0
                                dji = 1 if jket == ibra else 0
                                H[map[iket,jket],map[ibra,jbra]] += dii*djj - dij*dji
        return H



    def dHele(self, R):
        pass

    def dipole(self, R):
        pass

    def cap(self, R):
        pass


def Vnuc(R, params):
    aR = params["aR"]
    bR = params["bR"]
    return np.exp(-aR*R**2)/np.sqrt(R**2+bR)

def map_CI(xpts):
    # generate index mapping between single particle grid points and DVR slater determinants
    mapping = np.full((xpts, xpts), -1)
    k = 0
    for i in range(xpts):
        for j in range(i+1,xpts):
            mapping[i,j] = k
            k += 1

    return mapping


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
    # YOU'RE VERY BAD! ELECTRON MASS NOT NUCLEAR MASS!!!
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


