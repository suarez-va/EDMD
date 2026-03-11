import numpy as np

class FCILindbladCAP:
    def __init__(self, xi, wi, eta, ep, wpq, Pipq, Dipn, En, Wnm, Pinm):

        self.xi = xi
        self.wi = wi
        self.eta = eta
        self.ep = ep
        self.wpq = wpq
        self.Pipq = Pipq
        self.Dipn = Dipn
        self.En = En
        self.Wnm = Wnm
        self.Pinm = Pinm

        self.ndvr = self.xi.shape[0]
        self.nbo1 = self.ep.shape[0]
        self.nbo2 = self.En.shape[0]
        self.hpq = np.diag(self.ep) - 1j * self.eta * self.wpq
        self.Hnm = np.diag(self.En) - 1j * self.eta * self.Wnm

        self.N0 = 0j
        self.Ppq = np.zeros((self.nbo1, self.nbo1), dtype=np.complex128)
        self.Cn = np.zeros((self.nbo2), dtype=np.complex128)

    def kernel(self, timestep = 0.0025, nsteps = 250000, nprint = 1000):
        tpts = int(np.ceil(nsteps / nprint))
        t = timestep * nprint * np.arange(tpts)

        Px1t = np.zeros((self.ndvr, tpts), dtype=np.float64)
        Px2t = np.zeros((self.ndvr, tpts), dtype=np.float64)
        PE1t = np.zeros((self.nbo1, tpts), dtype=np.float64)
        PE2t = np.zeros((self.nbo2, tpts), dtype=np.float64)

        iprint = 0
        for i in range(nsteps):
            if i % nprint == 0:
                print(t[iprint])
                n0e = 2 * self.N0
                n1e = 2 * np.linalg.trace(self.Ppq)
                n2e = np.sum(self.Cn.conj()*self.Cn)
                print(f"n0e: {n0e}")
                print(f"n1e: {n1e}")
                print(f"n2e: {n2e}")
                print(f"total: {n0e + n1e + n2e}")
                Px1t[:,iprint] = 2 * np.tensordot(self.Pipq, self.Ppq, axes=([1,2],[1,0])).real
                Px2t[:,iprint] = (self.Pinm @ self.Cn @ self.Cn.conj()).real
                PE1t[:,iprint] = 2 * np.diag(self.Ppq).real
                PE2t[:,iprint] = (self.Cn.conj()*self.Cn).real
                iprint += 1

            DCip = np.tensordot(self.Dipn, self.Cn, axes=([2],[0])); Lpq = 2 * self.eta * DCip.conj().T * self.wi[None,:] @ DCip
            Nk1 = 2 * self.eta * np.sum(self.wi * np.tensordot(self.Pipq, self.Ppq, axes=([1,2],[1,0])))
            Pk1 = -1j * (self.hpq @ self.Ppq - self.Ppq @ self.hpq.conj().T) + Lpq
            Ck1 = -1j * self.Hnm @ self.Cn

            DCip = np.tensordot(self.Dipn, self.Cn + 0.5 * timestep * Ck1, axes=([2],[0])); Lpq = 2 * self.eta * DCip.conj().T * self.wi[None,:] @ DCip
            Nk2 = 2 * self.eta * np.sum(self.wi * np.tensordot(self.Pipq, self.Ppq + 0.5 * timestep * Pk1, axes=([1,2],[1,0])))
            Pk2 = -1j * (self.hpq @ (self.Ppq + 0.5 * timestep * Pk1) - (self.Ppq + 0.5 * timestep * Pk1) @ self.hpq.conj().T) + Lpq
            Ck2 = -1j * self.Hnm @ (self.Cn + 0.5 * timestep * Ck1)

            DCip = np.tensordot(self.Dipn, self.Cn + 0.5 * timestep * Ck2, axes=([2],[0])); Lpq = 2 * self.eta * DCip.conj().T * self.wi[None,:] @ DCip
            Nk3 = 2 * self.eta * np.sum(self.wi * np.tensordot(self.Pipq, self.Ppq + 0.5 * timestep * Pk2, axes=([1,2],[1,0])))
            Pk3 = -1j * (self.hpq @ (self.Ppq + 0.5 * timestep * Pk2) - (self.Ppq + 0.5 * timestep * Pk2) @ self.hpq.conj().T) + Lpq
            Ck3 = -1j * self.Hnm @ (self.Cn + 0.5 * timestep * Ck2)

            DCip = np.tensordot(self.Dipn, self.Cn + timestep * Ck3, axes=([2],[0])); Lpq = 2 * self.eta * DCip.conj().T * self.wi[None,:] @ DCip
            Nk4 = 2 * self.eta * np.sum(self.wi * np.tensordot(self.Pipq, self.Ppq + timestep * Pk3, axes=([1,2],[1,0])))
            Pk4 = -1j * (self.hpq @ (self.Ppq + timestep * Pk3) - (self.Ppq + timestep * Pk3) @ self.hpq.conj().T) + Lpq
            Ck4 = -1j * self.Hnm @ (self.Cn + timestep * Ck3)

            self.N0 += timestep / 6.0 * (Nk1 + 2 * Nk2 + 2 * Nk3 + Nk4)
            self.Ppq += timestep / 6.0 * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4)
            self.Cn += timestep / 6.0 * (Ck1 + 2 * Ck2 + 2 * Ck3 + Ck4)

        np.savez('Lindblad', t=t, xi=self.xi, Px1t=Px1t, Px2t=Px2t, ep=self.ep, PE1t=PE1t, En=self.En, PE2t=PE2t)

        return self
