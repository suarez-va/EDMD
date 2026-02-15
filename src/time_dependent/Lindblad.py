import numpy as np

class LindbladCAP:
    def __init__(self, eta, wi, Ek, Dikn, En, Wnm, Cn0):

        self.eta = eta
        self.wi = wi
        self.Ek = Ek
        self.Dikn = Dikn
        self.En = En
        self.Wnm = Wnm

        self.nxdvr = self.wi.shape[0]
        self.nk = self.Ek.shape[0]
        self.nn = self.En.shape[0]
        self.Hnm = np.diag(self.En) - 1j * self.eta * self.Wnm

        self.Cn = Cn0
        self.Pkl = np.zeros((self.nk, self.nk), dtype=np.complex128)

    def kernel(self, timestep = 0.0025, nsteps = 250000, nprint = 1000):
        tpts = int(np.ceil(nsteps / nprint))
        t = timestep * nprint * np.arange(tpts)

        iprint = 0
        for i in range(nsteps):
            if i % nprint == 0:
                print(t[iprint])
                Nk = 2 * np.linalg.trace(self.Pkl)
                Nn = np.sum(self.Cn.conj() * self.Cn)
                print(f"Nk: {Nk}")
                print(f"Nn: {Nn}")
                print(f"total: {Nk + Nn}")
                iprint += 1

            Dik = np.tensordot(self.Dikn, self.Cn, axes=([2],[0])); Lkl = 2 * self.eta * ((Dik.T * self.wi[None,:]) @ Dik.conj())
            Ck1 = -1j * self.En * self.Cn - self.eta * (self.Wnm @ self.Cn)
            Pk1 = -1j * (self.Ek[:,None] - self.Ek[None,:]) * self.Pkl + Lkl

            Dik = np.tensordot(self.Dikn, self.Cn + 0.5 * timestep * Ck1, axes=([2],[0])); Lkl = 2 * self.eta * ((Dik.T * self.wi[None,:]) @ Dik.conj())
            Ck2 = -1j * self.En * (self.Cn + 0.5 * timestep * Ck1) - self.eta * (self.Wnm @ (self.Cn + 0.5 * timestep * Ck1))
            Pk2 = -1j * (self.Ek[:,None] - self.Ek[None,:]) * (self.Pkl + 0.5 * timestep * Pk1) + Lkl

            Dik = np.tensordot(self.Dikn, self.Cn + 0.5 * timestep * Ck2, axes=([2],[0])); Lkl = 2 * self.eta * ((Dik.T * self.wi[None,:]) @ Dik.conj())
            Ck3 = -1j * self.En * (self.Cn + 0.5 * timestep * Ck2) - self.eta * (self.Wnm @ (self.Cn + 0.5 * timestep * Ck2))
            Pk3 = -1j * (self.Ek[:,None] - self.Ek[None,:]) * (self.Pkl + 0.5 * timestep * Pk2) + Lkl

            Dik = np.tensordot(self.Dikn, self.Cn + timestep * Ck3, axes=([2],[0])); Lkl = 2 * self.eta * ((Dik.T * self.wi[None,:]) @ Dik.conj())
            Ck4 = -1j * self.En * (self.Cn + timestep * Ck3) - self.eta * (self.Wnm @ (self.Cn + timestep * Ck3))
            Pk4 = -1j * (self.Ek[:,None] - self.Ek[None,:]) * (self.Pkl + timestep * Pk3) + Lkl

            self.Pkl += timestep / 6.0 * (Pk1 + 2 * Pk2 + 2 * Pk3 + Pk4)
            self.Cn += timestep / 6.0 * (Ck1 + 2 * Ck2 + 2 * Ck3 + Ck4)
        
        np.savez('Lindblad', t=t)
        return self
