import os
import numpy as np
import time

class EDMD:
    def __init__(self, model, eta, wk, ep, cip, wpq, En, Cijn, Wnm):

        self.model = model
        self.eta = eta
        self.wk = wk
        self.ep = ep
        self.cip = cip
        self.wpq = wpq
        self.En = En
        self.Cijn = Cijn
        self.Wnm = Wnm

        self.ndvr = self.wk.shape[0]
        self.nmo = self.ep.shape[0]
        self.nbo = self.En.shape[0]
        self.hpq = np.diag(self.ep) - 1j * self.eta * self.wpq
        self.Hnm = np.diag(self.En) - 1j * self.eta * self.Wnm
        self.Dipn = np.einsum('ijn,jp->ipn', self.Cijn, self.cip.conj())

    def kernel(self, Cn0, timestep = 0.0025, nsteps = 250000, nprint = 1000):
        n0 = 0j
        rpq = np.zeros((self.nmo, self.nmo), dtype=np.complex128)
        Cn = Cn0

        tpts = int(np.ceil(nsteps / nprint))
        t = timestep * nprint * np.arange(tpts)

        n1t = np.zeros((self.ndvr, tpts), dtype=np.float64)
        n2t = np.zeros((self.ndvr, tpts), dtype=np.float64)

        iprint = 0
        for i in range(nsteps):
            if i % nprint == 0:
                print(t[iprint])
                n0e = 2 * n0
                n1e = 2 * np.linalg.trace(rpq)
                n2e = np.sum(Cn.conj()*Cn)
                print(f"n0e: {n0e}")
                print(f"n1e: {n1e}")
                print(f"n2e: {n2e}")
                print(f"total: {n0e + n1e + n2e}")
                Cijt = np.sum(self.Cijn * Cn[None,None,:], axis=2)
                n1t[:,iprint] = 2 * np.diag(np.matmul(self.cip, np.matmul(rpq, self.cip.conj().T))).real
                n2t[:,iprint] = 2 * np.sum(Cijt.conj() * Cijt, axis=1).real
                iprint += 1
        
            #time1 = time.time()
            DCip = np.tensordot(self.Dipn, Cn, axes=([2],[0])); Lpq = 2 * self.eta * np.matmul(DCip.conj().T * self.wk[None,:], DCip)
            #time2 = time.time()
            nk1 = 2 * self.eta * np.sum(self.wk * np.diag(np.matmul(self.cip, np.matmul(rpq, self.cip.conj().T))))
            #time3 = time.time()
            rk1 = -1j * (np.matmul(self.hpq, rpq) - np.matmul(rpq, self.hpq.conj().T)) + Lpq
            #time4 = time.time()
            Ck1 = -1j * np.matmul(self.Hnm, Cn)
            #time5 = time.time()
            #print(f'DCip: {time1p5 - time1}')
            #print(f'Lpq: {time2 - time1p5}')
            #print(f'nk1: {time3 - time2}')
            #print(f'rk1: {time4 - time3}')
            #print(f'Ck1: {time5 - time4}')
            
            DCip = np.tensordot(self.Dipn, Cn + 0.5 * timestep * Ck1, axes=([2],[0])); Lpq = 2 * self.eta * np.matmul(DCip.conj().T * self.wk[None,:], DCip)
            nk2 = 2 * self.eta * np.sum(self.wk * np.diag(np.matmul(self.cip, np.matmul(rpq + 0.5 * timestep * rk1, self.cip.conj().T))))
            rk2 = -1j * (np.matmul(self.hpq, rpq + 0.5 * timestep * rk1) - np.matmul(rpq + 0.5 * timestep * rk1, self.hpq.conj().T)) + Lpq
            Ck2 = -1j * np.matmul(self.Hnm, Cn + 0.5 * timestep * Ck1)
           
            DCip = np.tensordot(self.Dipn, Cn + 0.5 * timestep * Ck2, axes=([2],[0])); Lpq = 2 * self.eta * np.matmul(DCip.conj().T * self.wk[None,:], DCip)
            nk3 = 2 * self.eta * np.sum(self.wk * np.diag(np.matmul(self.cip, np.matmul(rpq + 0.5 * timestep * rk2, self.cip.conj().T))))
            rk3 = -1j * (np.matmul(self.hpq, rpq + 0.5 * timestep * rk2) - np.matmul(rpq + 0.5 * timestep * rk2, self.hpq.conj().T)) + Lpq
            Ck3 = -1j * np.matmul(self.Hnm, Cn + 0.5 * timestep * Ck2)

            DCip = np.tensordot(self.Dipn, Cn + timestep * Ck3, axes=([2],[0])); Lpq = 2 * self.eta * np.matmul(DCip.conj().T * self.wk[None,:], DCip)
            nk4 = 2 * self.eta * np.sum(self.wk * np.diag(np.matmul(self.cip, np.matmul(rpq + timestep * rk3, self.cip.conj().T))))
            rk4 = -1j * (np.matmul(self.hpq, rpq + timestep * rk3) - np.matmul(rpq + timestep * rk3, self.hpq.conj().T)) + Lpq
            Ck4 = -1j * np.matmul(self.Hnm, Cn + timestep * Ck3)
        
            n0 += timestep / 6.0 * (nk1 + 2 * nk2 + 2 * nk3 + nk4)
            rpq += timestep / 6.0 * (rk1 + 2 * rk2 + 2 * rk3 + rk4)
            Cn += timestep / 6.0 * (Ck1 + 2 * Ck2 + 2 * Ck3 + Ck4)
        
        np.savez('Lindblad', t=t, xi=self.model.xi(), n1t=n1t, n2t=n2t)

        return self
