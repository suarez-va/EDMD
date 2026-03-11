import os
from typing import Optional
import itertools
import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import LinearOperator, eigsh, bicg, bicgstab, cgs, gmres, minres, lgmres, qmr, gcrotmk, tfqmr
from dvr_basis.colbert_miller_dvr import dvr_x, dvr_p, dvr_T, dvr_wi
from model_systems.models import Model
import time

def validate_nele_spin(nele: int, spin: str, Sz: Optional[float] = None):
    allowed_cases = {
        1: [['doublet', 0.5], ['doublet', -0.5]],
        2: [['singlet', 0.0], ['triplet', 1.0], ['triplet', 0.0], ['triplet', -1.0]],
        3: [['doublet', 0.5], ['doublet', -0.5], ['quartet', 1.5], ['quartet', 0.5], ['quartet', -0.5], ['quartet', -1.5]]
    }
    if nele not in allowed_cases:
        raise ValueError(f"FCIDVR not implemented for nele = {nele}")
    if Sz is None:
        allowed_spin = {case[0] for case in allowed_cases[nele]}
        if spin not in allowed_spin:
            raise ValueError(f"Invalid spin = '{spin}' for nele = {nele}.\n Allowed spin: {allowed_spin}")
    elif [spin, Sz] not in allowed_cases[nele]:
        raise ValueError(f"Invalid spin = '{spin}' and projection Sz = {Sz} for nele = {nele}.\n Allowed [spin, Sz]: {allowed_cases[nele]}")
    return None

def fci_mapping(ndvr: int, nele: int, spin: str):
    validate_nele_spin(nele, spin)
    if nele == 1:
        if spin == 'doublet':
            nfci = ndvr
            map_norm = np.ones((ndvr), dtype = np.float64)
            map_idx = np.arange(ndvr, dtype = int)
            return nfci, map_norm, map_idx
    if nele == 2:
        if spin == 'singlet':
            nfci = int((ndvr + 1) * ndvr / 2)
            map_norm = np.triu((1 - (1 - 1 / np.sqrt(2)) * np.eye(ndvr)), k = 0)
            map_idx = np.triu_indices(ndvr, k = 0)
            return nfci, map_norm, map_idx
        if spin == 'triplet':
            nfci = int(ndvr * (ndvr - 1) / 2)
            map_norm = np.triu(np.ones((ndvr, ndvr)), k = 1)
            map_idx = np.triu_indices(ndvr, k = 1)
            return nfci, map_norm, map_idx
    if nele == 3:
        if spin == 'doublet':
            nfci = int((ndvr + 1) * ndvr * (ndvr - 1) / 3)
            i, j, k = np.indices((ndvr, ndvr, ndvr))
            map_norm1 = np.zeros((ndvr, ndvr, ndvr)); map_norm1[(i < j) & (j < k)] = 1.0; map_norm1[(i < j) & (j == k)] = np.sqrt(0.5)
            map_norm2 = np.zeros((ndvr, ndvr, ndvr)); map_norm2[(i < j) & (j < k)] = 1.0; map_norm2[(i == j) & (j < k)] = np.sqrt(2.0/3.0)
            map_norm = (map_norm1, map_norm2)
            map_idx1 = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if (i < j < k) or (i < j and j == k)]))
            map_idx2 = tuple(np.array(x, dtype = int) for x in zip(*[(i, j, k) for (i, j, k) in itertools.combinations_with_replacement(range(ndvr), 3) if (i < j < k) or (i == j and j < k)]))
            map_idx = (map_idx1, map_idx2)
            return nfci, map_norm, map_idx
    assert False

def fci_operator(ndvr: int, nele: int, spin: str, hij: np.ndarray, gik: Optional[np.ndarray] = None):
    validate_nele_spin(nele, spin)
    nfci, map_norm, map_idx = fci_mapping(ndvr, nele, spin)
    gik = np.zeros((ndvr, ndvr)) if gik is None else gik
    matvec = lambda C: C
    if nele == 1:
        if spin == 'doublet':
            def matvec(C):
                Cj = np.zeros((ndvr), dtype = np.complex128); Cj[map_idx] = C; nCj = map_norm * Cj
                Cj = nCj
                hC = hij @ Cj
                Ci = map_norm * hC
                return Ci[map_idx]
    if nele == 2:
        if spin == 'singlet':
            def matvec(C):
                Cjl = np.zeros((ndvr, ndvr), dtype = np.complex128); Cjl[map_idx] = C; nCjl = map_norm * Cjl
                gC = gik * Cjl
                Cjl = nCjl + nCjl.swapaxes(0,1)
                hC = hij @ Cjl
                Cik = map_norm * (hC + hC.swapaxes(0,1)) + gC
                return Cik[map_idx]
        if spin == 'triplet':
            def matvec(C):
                Cjl = np.zeros((ndvr, ndvr), dtype = np.complex128); Cjl[map_idx] = C; nCjl = map_norm * Cjl
                gC = gik * Cjl
                Cjl = nCjl - nCjl.swapaxes(0,1)
                hC = hij @ Cjl
                Cik = map_norm * (hC - hC.swapaxes(0,1)) + gC
                return Cik[map_idx]
    if nele == 3:
        gikm = gik[:,:,None] + gik[:,None,:] + gik[None,:,:]
        if spin == 'doublet':
            def matvec(C):
                Cjln1 = np.zeros((ndvr, ndvr, ndvr), dtype = np.complex128); Cjln1[map_idx[0]] = C[:int(nfci/2)]; nCjln1 = map_norm[0] * Cjln1
                Cjln2 = np.zeros((ndvr, ndvr, ndvr), dtype = np.complex128); Cjln2[map_idx[1]] = C[int(nfci/2):]; nCjln2 = map_norm[1] * Cjln2
                gC1 = gikm*Cjln1; gC2 = gikm*Cjln2
                Cjln1 = nCjln1 + nCjln1.swapaxes(1,2) - 0.5*(nCjln1.swapaxes(0,1) + nCjln1.transpose(2,0,1) + nCjln1.transpose(1,2,0) + nCjln1.swapaxes(0,2)) + 0.5*np.sqrt(3)*(nCjln2.swapaxes(0,1) - nCjln2.transpose(2,0,1) + nCjln2.transpose(1,2,0) - nCjln2.swapaxes(0,2))
                Cjln2 = nCjln2 - nCjln2.swapaxes(1,2) + 0.5*(nCjln2.swapaxes(0,1) - nCjln2.transpose(2,0,1) - nCjln2.transpose(1,2,0) + nCjln2.swapaxes(0,2)) + 0.5*np.sqrt(3)*(nCjln1.swapaxes(0,1) + nCjln1.transpose(2,0,1) - nCjln1.transpose(1,2,0) - nCjln1.swapaxes(0,2))
                hC1 = (hij @ Cjln1.reshape(ndvr, ndvr**2)).reshape(ndvr, ndvr, ndvr); Ch1 = (Cjln1.reshape(ndvr**2, ndvr) @ hij.conj()).reshape(ndvr, ndvr, ndvr)
                hC2 = (hij @ Cjln2.reshape(ndvr, ndvr**2)).reshape(ndvr, ndvr, ndvr); Ch2 = (Cjln2.reshape(ndvr**2, ndvr) @ hij.conj()).reshape(ndvr, ndvr, ndvr)
                Cikm1 = map_norm[0]*(hC1 + Ch1.swapaxes(1,2) + Ch1) + gC1
                Cikm2 = map_norm[1]*(hC2 - Ch2.swapaxes(1,2) + Ch2) + gC2
                return np.concatenate((Cikm1[map_idx[0]], Cikm2[map_idx[1]]))

    def matmat(Cm):
        m = Cm.shape[1]
        Cn = np.zeros_like(Cm, dtype=np.complex128)
        for j in range(m):
            Cn[:, j] = matvec(Cm[:, j])
        return Cn
    return LinearOperator(shape=(nfci, nfci), matvec=matvec, matmat=matmat, dtype=np.complex128)

def fci_wfn(ndvr: int, nele: int, spin: str, Sz: float, C: np.ndarray):
    validate_nele_spin(nele, spin, Sz)
    nfci, map_norm, map_idx = fci_mapping(ndvr, nele, spin)
    wfn = np.array([0], dtype=np.complex128)
    if nele == 1:
        wfn = np.zeros((ndvr, 2), dtype=np.complex128)
        if spin == 'doublet':
            Ci = np.zeros((ndvr), dtype = np.complex128); Ci[map_idx] = C; nCi = map_norm * Ci
            Ci = nCi
            if Sz == 0.5:
                wfn[:,0] += Ci
            if Sz == -0.5:
                wfn[:,1] += Ci
    if nele == 2:
        wfn = np.zeros((ndvr, 2, ndvr, 2), dtype=np.complex128)
        if spin == 'singlet':
            Cij = np.zeros((ndvr, ndvr), dtype = np.complex128); Cij[map_idx] = C; nCij = map_norm * Cij
            Cij = 1/np.sqrt(2)*(nCij + nCij.swapaxes(0,1))
            if Sz == 0.0:
                wfn[:,0,:,1] += 1/np.sqrt(2)*Cij
                wfn[:,1,:,0] += -1/np.sqrt(2)*Cij
        if spin == 'triplet':
            Cij = np.zeros((ndvr, ndvr), dtype = np.complex128); Cij[map_idx] = C; nCij = map_norm * Cij
            Cij = 1/np.sqrt(2)*(nCij - nCij.swapaxes(0,1))
            if Sz == 1.0:
                wfn[:,0,:,0] += Cij
            if Sz == 0.0:
                wfn[:,0,:,1] += 1/np.sqrt(2)*Cij
                wfn[:,1,:,0] += 1/np.sqrt(2)*Cij
            if Sz == -1.0:
                wfn[:,1,:,1] += Cij
    if nele == 3:
        wfn = np.zeros((ndvr, 2, ndvr, 2, ndvr, 2), dtype=np.complex128)
        if spin == 'doublet':
            Cijk1 = np.zeros((ndvr, ndvr, ndvr), dtype = np.complex128); Cijk1[map_idx[0]] = C[:int(nfci/2)]; nCijk1 = map_norm[0] * Cijk1
            Cijk2 = np.zeros((ndvr, ndvr, ndvr), dtype = np.complex128); Cijk2[map_idx[1]] = C[int(nfci/2):]; nCijk2 = map_norm[1] * Cijk2
            Cijk = (1/np.sqrt(2)*(nCijk1 + nCijk1.swapaxes(1,2) - nCijk1.swapaxes(0,1) - nCijk1.transpose(2,0,1))
                + 1/np.sqrt(6)*(-nCijk2 + nCijk2.swapaxes(1,2) + nCijk2.swapaxes(0,1) - nCijk2.transpose(2,0,1))
                + np.sqrt(2/3)*(nCijk2.transpose(1,2,0) - nCijk2.swapaxes(0,2)))
            if Sz == 0.5:
                wfn[:,0,:,0,:,1] += 1/np.sqrt(6)*Cijk
                wfn[:,0,:,1,:,0] += -1/np.sqrt(6)*Cijk.swapaxes(1,2)
                wfn[:,1,:,0,:,0] += -1/np.sqrt(6)*Cijk.swapaxes(0,2)
            if Sz == -0.5:
                wfn[:,0,:,1,:,1] += -1/np.sqrt(6)*Cijk.swapaxes(0,2)
                wfn[:,1,:,0,:,1] += -1/np.sqrt(6)*Cijk.swapaxes(1,2)
                wfn[:,1,:,1,:,0] += 1/np.sqrt(6)*Cijk
    return wfn

def fci_density(ndvr: int, nele: int, spinbra: str, Szbra: float, Cbra: np.ndarray, spinket: Optional[str] = None, Szket: Optional[float] = None, Cket: Optional[np.ndarray] = None):
    spinket = spinbra if spinket is None else spinket
    Szket = Szbra if Szket is None else Szket
    Cket = Cbra if Cket is None else Cket
    bra = fci_wfn(ndvr, nele, spinbra, Szbra, Cbra)
    ket = fci_wfn(ndvr, nele, spinket, Szket, Cket)
    rho = nele*(bra.conj() * ket).sum(axis=tuple(range(2, int(2*nele))))
    return rho

def fci_1rdm(ndvr: int, nele: int, spinbra: str, Szbra: float, Cbra: np.ndarray, spinket: Optional[str] = None, Szket: Optional[float] = None, Cket: Optional[np.ndarray] = None):
    spinket = spinbra if spinket is None else spinket
    Szket = Szbra if Szket is None else Szket
    Cket = Cbra if Cket is None else Cket
    gam = np.zeros((ndvr, 2, ndvr, 2), dtype=np.complex128)
    bra = fci_wfn(ndvr, nele, spinbra, Szbra, Cbra)
    ket = fci_wfn(ndvr, nele, spinket, Szket, Cket)
    bra_tmp = bra.reshape(ndvr*2, (ndvr*2)**(nele-1))
    ket_tmp = ket.reshape(ndvr*2, (ndvr*2)**(nele-1))
    gam_tmp = nele*(bra_tmp.conj() @ ket_tmp.T).T
    gam = gam_tmp.reshape(ndvr, 2, ndvr, 2)
    return gam

def fci_dyson(ndvr: int, nele: int, spinbra: str, Szbra: float, Cbra: np.ndarray, spinket: str, Szket: float, Cket: np.ndarray):
    bra = fci_wfn(ndvr, nele-1, spinbra, Szbra, Cbra)
    ket = fci_wfn(ndvr, nele, spinket, Szket, Cket)
    Dis = np.sqrt(nele)*(bra.conj()[None,None,...] * ket).sum(axis=tuple(range(2, int(2*nele))))
    return Dis

#def fci_1rdm(ndvr: int, nele: int, spin: str, Sz: float, C: np.ndarray):
#    gam = np.zeros((ndvr, 2, ndvr, 2), dtype=np.complex128)
#    wfn = fci_wfn(ndvr, nele, spin, Sz, C)
#    wfn_tmp = wfn.reshape(ndvr*2, (ndvr*2)**(nele-1))
#    gam_tmp = nele*(wfn_tmp.conj() @ wfn_tmp.T).T
#    gam = gam_tmp.reshape(ndvr, 2, ndvr, 2)
#    return gam

#def fci_dyson(ndvr: int, nele1: int, spin1: str, Sz1: float, C1: np.ndarray, nele2: int, spin2: str, Sz2: float, C2: np.ndarray):
#    assert nele2 - nele1 == 1, f"CI vector C1 must be the N-1 electron state while CI vector C2 must be the N electron state."
#    validate_nele_spin(nele1, spin1, Sz1)
#    validate_nele_spin(nele2, spin2, Sz2)
#    wfn1 = fci_wfn(ndvr, nele1, spin1, Sz1, C1)
#    wfn2 = fci_wfn(ndvr, nele2, spin2, Sz2, C2)
#    Dis = np.sqrt(nele2)*(wfn1.conj()[None,None,...] * wfn2).sum(axis=tuple(range(2, int(2*nele2))))
#    return Dis

class FCIDVR:
    def __init__(self, model: Model, xa: float, xb: float, xN: int, xbounds: str):
        self.model = model
        self.xa = xa
        self.xb = xb
        self.xN = xN
        self.xbounds = xbounds
        assert self.xbounds in ('(a,b)', '(0,inf)', '(-inf,inf)'), f"Colber Miller Electronic DVR Bounds: {self.xbounds}, must be '(a,b)', '(0,inf)', or '(-inf,inf)'."

        self.nxdvr = int(self.xN - 1) if self.xbounds=='(a,b)' else int(self.xN) if self.xbounds=='(0,inf)' else int(self.xN + 1) if self.xbounds=='(-inf,inf)' else 0

    def xi(self) -> np.ndarray:
        return np.linspace(self.xa, self.xb, self.nxdvr)

    def pij(self) -> np.ndarray:
        return dvr_p(self.xa, self.xb, self.xN, self.xbounds)

    # generate hcore using Colbert-Miller syle DVR
    def hij(self, R: float) -> np.ndarray:
        return dvr_T(1, self.xa, self.xb, self.xN, self.xbounds) + np.diag(self.model.VeR(self.xi(), R))

    # generate derivative of hcore with respect to R using Colbert-Miller syle DVR for kinetic energy
    def d1hij(self, R: float) -> np.ndarray:
        return np.diag(self.model.d1VeR(self.xi(), R))

    def d2hij(self, R: float) -> np.ndarray:
        return np.diag(self.model.d2VeR(self.xi(), R))

    def gik(self) -> np.ndarray:
        xi=self.xi()
        #return 0.0*self.model.Vee(xi[:,None],xi[None,:])
        return self.model.Vee(xi[:,None],xi[None,:])

    def kernel(self, R: float, nele: int = 1, spin: str = 'doublet', nbo: int = 100, derivative_order: int = 2):
        validate_nele_spin(nele, spin)
        assert derivative_order in (0, 1, 2), f"derivative order: {derivative_order}, must be 0, 1, or 2."
        xi = self.xi()
        pij = self.pij()
        hij = self.hij(R)
        gik = self.gik()
        X = fci_operator(self.nxdvr, nele, spin, np.diag(xi))
        P = fci_operator(self.nxdvr, nele, spin, pij)
        H = fci_operator(self.nxdvr, nele, spin, hij, gik)
        d1H = fci_operator(self.nxdvr, nele, spin, self.d1hij(R))
        d2H = fci_operator(self.nxdvr, nele, spin, self.d2hij(R))
        nfci = H.shape[0]

        # \mathbf{H}(R) \mathbf{C}_n(R) = E_n(R) \mathbf{C}_n(R)
        if (nbo >= nfci - 1):
            En, Cn = eigh(H.matmat(np.eye(nfci)), subset_by_index = (0, nbo - 1))
        else:
            En, Cn = eigsh(H, k = nbo, which = 'SA')
            idx = np.argsort(En)
            En = En[idx]; Cn = Cn[:,idx]
        Cn *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,:])

        d1En = np.zeros_like(En); d1Cn = np.zeros_like(Cn)
        d2En = np.zeros_like(En); d2Cn = np.zeros_like(Cn)
        if (derivative_order >= 1):
            d1HCn = d1H.matmat(Cn); d2HCn = d2H.matmat(Cn)
            for n in range(nbo):
                print(n)
                A_matvec = lambda x: H.matvec(x) - En[n] * x
                A = LinearOperator(shape=(nfci, nfci), matvec=A_matvec, rmatvec=A_matvec, dtype=np.complex128)
                def invA_matvec(x):
                    C_DE = Cn*(1 - np.eye(nbo)[None,:,n])/(En[None,:] - En[n] + np.eye(nbo)[None,:,n])
                    Cx = Cn.conj().T @ x
                    return np.einsum('m,Im->I', Cx, C_DE)

                d1En[n] = np.dot(Cn[:,n].conj(), d1HCn[:,n]).real
                b = -(d1HCn[:,n] - d1En[n]*Cn[:,n])
                x0 = invA_matvec(b)
                d1C, info = bicg(A, b, x0, rtol=1e-9)
                d1Cn[:,n] = d1C - Cn[:,n]*np.dot(Cn[:,n].conj(), d1C)

                if (derivative_order == 2):
                    d2En[n] = np.dot(Cn[:,n].conj(), d2HCn[:,n]).real + 2*np.dot(d1Cn[:,n].conj(), d1HCn[:,n]).real
                    b = -(d2HCn[:,n] - d2En[n]*Cn[:,n]) - 2*(d1H.matvec(d1Cn[:,n]) - d1En[n]*d1Cn[:,n])
                    x0 = invA_matvec(b)
                    d2C, info = bicg(A, b, x0, rtol=1e-9)
                    d2Cn[:,n] = d2C - Cn[:,n]*(np.dot(Cn[:,n].conj(), d2C) + np.dot(d1Cn[:,n].conj(), d1Cn[:,n]))

        nac01 = Cn.conj().T @ d1Cn
        nac11 = d1Cn.conj().T @ d1Cn
        nac02 = Cn.conj().T @ d2Cn

        xnm = Cn.conj().T @ X.matmat(Cn)
        pnm = Cn.conj().T @ P.matmat(Cn)
        En += self.model.VR(R)
        d1En += self.model.d1VR(R)
        d2En += self.model.d2VR(R)

        np.savez(f'fcidvr_{nele}ele_{spin}', xi=xi, xnm=xnm, pnm=pnm, En=En, Cn=Cn, d1En=d1En, d1Cn=d1Cn, d2En=d2En, d2Cn=d2Cn, nac01=nac01, nac11=nac11, nac02=nac02)
        return self

def compute_density(fcidvr_file: str, Sz: float):
    if not os.path.isfile(fcidvr_file):
        raise FileNotFoundError(f"File not found: {fcidvr_file}")

    base = fcidvr_file.removesuffix(".npz")
    parts = base.split("_")
    nele = int(parts[1][0])
    spin = parts[2]

    fcidvr_data = np.load(fcidvr_file)
    xi = fcidvr_data['xi']
    Cn = fcidvr_data['Cn']
    ndvr = xi.shape[0]
    nbo = Cn.shape[1]
    Pisnm = np.zeros((ndvr, 2, nbo, nbo), dtype=np.complex128)

    for n in range(nbo):
        print(n)
        for m in range(nbo):
            Pisnm[:,:,n,m] = fci_density(ndvr, nele, spin, Sz, Cbra = Cn[:,n], Cket = Cn[:,m])
            #Pisnm[:,:,n,m] = np.einsum('iaia->ia', fci_1rdm(ndvr, nele, spin, Sz, Cbra = Cn[:,n], Cket = Cn[:,m]))

    np.savez(f'fcidvr_{nele}ele_{spin}_{Sz}Sz_density', xi=xi, Pisnm=Pisnm)
    return None

def compute_dyson(fcidvr_file_bra: str, Szbra: float, fcidvr_file_ket: str, Szket: float):
    if not os.path.isfile(fcidvr_file_bra):
        raise FileNotFoundError(f"N-1 electron file not found: {fcidvr_file_bra}")
    if not os.path.isfile(fcidvr_file_ket):
        raise FileNotFoundError(f"N electron file not found: {fcidvr_file_ket}")

    basebra = fcidvr_file_bra.removesuffix(".npz"); baseket = fcidvr_file_ket.removesuffix(".npz")
    partsbra = basebra.split("_"); partsket = baseket.split("_")
    nelebra = int(partsbra[1][0]); neleket = int(partsket[1][0])
    spinbra = partsbra[2]; spinket = partsket[2]

    assert neleket - nelebra == 1, f"fcidvr_file_ket must be the N electron file while fcidvr_file_bra must be the N-1 electron file."
    validate_nele_spin(nelebra, spinbra, Szbra); validate_nele_spin(neleket, spinket, Szket)

    fcidvr_data_bra = np.load(fcidvr_file_bra)
    fcidvr_data_ket = np.load(fcidvr_file_ket)
    xi = fcidvr_data_bra['xi']
    Cbra = fcidvr_data_bra['Cn']; Cket = fcidvr_data_ket['Cn']
    ndvr = xi.shape[0]; nele = neleket 
    nbobra = Cbra.shape[1]; nboket = Cket.shape[1]
    Dispn = np.zeros((ndvr, 2, nbobra, nboket), dtype=np.complex128)

    for p in range(nbobra):
        print(f'p={p}')
        for n in range(nboket):
            Dispn[:,:,p,n] = fci_dyson(ndvr, nele, spinbra, Szbra, Cbra[:,p], spinket, Szket, Cket[:,n])

    np.savez(f'fcidvr_{nelebra}ele_{spinbra}_{Szbra}Sz_{neleket}ele_{spinket}_{Szket}Sz_dyson', xi=xi, Dispn=Dispn)
    return None

def compute_cap(fcidvr_file: str, acap: float, bcap: float, ncap: int = 2):
    if not os.path.isfile(fcidvr_file):
        raise FileNotFoundError(f"File not found: {fcidvr_file}")

    base = fcidvr_file.removesuffix(".npz")
    parts = base.split("_")
    nele = int(parts[1][0])
    spin = parts[2]

    fcidvr_data = np.load(fcidvr_file)
    xi = fcidvr_data['xi']
    Cn = fcidvr_data['Cn']
    ndvr = xi.shape[0]
    wi = dvr_wi(xi, acap, bcap, ncap)
    W = fci_operator(ndvr, nele, spin, np.diag(wi))
    Wnm = Cn.conj().T @ W.matmat(Cn)

    np.savez(base + '_cap', wi=wi, Wnm=Wnm)
    return None


#def compute_1rdm(fcidvr_file: str, Sz: float):
#    if not os.path.isfile(fcidvr_file):
#        raise FileNotFoundError(f"File not found: {fcidvr_file}")
#
#    base = fcidvr_file.removesuffix(".npz")
#    parts = base.split("_")
#    nele = int(parts[1][0])
#    spin = parts[2]
#
#    validate_nele_spin(nele, spin, Sz)
#
#    fcidvr_data = np.load(fcidvr_file)
#    xi = fcidvr_data['xi']
#    Cn = fcidvr_data['Cn']
#    ndvr = xi.shape[0]
#    nbo = Cn.shape[1]
#    gamn = np.zeros((ndvr, 2, ndvr, 2, nbo), dtype=np.complex128)
#
#    for n in range(nbo):
#        gamn[:,:,:,:,n] = fci_1rdm(ndvr, nele, spin, Sz, Cn[:,n])
#
#    np.savez(f'fcidvr_{nele}ele_{spin}_{Sz}Sz_1rdm', xi=xi, gamn=gamn)
#
#    return None
#
#def compute_rho(fcidvr_file: str, Sz: float):
#    if not os.path.isfile(fcidvr_file):
#        raise FileNotFoundError(f"File not found: {fcidvr_file}")
#
#    base = fcidvr_file.removesuffix(".npz")
#    parts = base.split("_")
#    nele = int(parts[1][0])
#    spin = parts[2]
#
#    validate_nele_spin(nele, spin, Sz)
#
#    fcidvr_data = np.load(fcidvr_file)
#    xi = fcidvr_data['xi']
#    Cn = fcidvr_data['Cn']
#    ndvr = xi.shape[0]
#    nbo = Cn.shape[1]
#    rhon = np.zeros((ndvr, 2, nbo), dtype=np.complex128)
#
#    for n in range(nbo):
#        rhon[:,:,n] = np.einsum('iaia->ia', fci_1rdm(ndvr, nele, spin, Sz, Cn[:,n]))
#
#    np.savez(f'fcidvr_{nele}ele_{spin}_{Sz}Sz_1rdm', xi=xi, rhon=rhon)
#
#    return None
#
#def compute_dyson(fcidvr_file1: str, Sz1: float, fcidvr_file2: str, Sz2: float):
#    if not os.path.isfile(fcidvr_file1):
#        raise FileNotFoundError(f"N-1 electron file not found: {fcidvr_file1}")
#    if not os.path.isfile(fcidvr_file2):
#        raise FileNotFoundError(f"N electron file not found: {fcidvr_file2}")
#
#    base1 = fcidvr_file1.removesuffix(".npz"); base2 = fcidvr_file2.removesuffix(".npz")
#    parts1 = base1.split("_"); parts2 = base2.split("_")
#    nele1 = int(parts1[1][0]); nele2 = int(parts2[1][0])
#    spin1 = parts1[2]; spin2 = parts2[2]
#
#    assert nele2 - nele1 == 1, f"fcidvr_file2 must be the N electron file while fcidvr_file1 must be the N-1 electron file."
#    validate_nele_spin(nele1, spin1, Sz1); validate_nele_spin(nele2, spin2, Sz2)
#
#    fcidvr_data1 = np.load(fcidvr_file1)
#    fcidvr_data2 = np.load(fcidvr_file2)
#    xi = fcidvr_data1['xi']
#    Cn1 = fcidvr_data1['Cn']; Cn2 = fcidvr_data2['Cn']
#    ndvr = xi.shape[0]
#    nbo1 = Cn1.shape[1]; nbo2 = Cn2.shape[1]
#    Dpn = np.zeros((ndvr, 2, nbo1, nbo2), dtype=np.complex128)
#
#    for p in range(nbo1):
#        for n in range(nbo2):
#            Dpn[:,:,p,n] = fci_dyson(ndvr, nele1, spin1, Sz1, Cn1[:,p], nele2, spin2, Sz2, Cn2[:,n])
#
#    np.savez(f'fcidvr_{nele1}ele_{spin1}_{Sz1}Sz_{nele2}ele_{spin2}_{Sz2}Sz_dyson', xi=xi, Dpn=Dpn)
#
#    return None


#def compute_dyson(fcidvr_file1: str, Sz1: float, fcidvr_file2: str, Sz2: float):
#    if not os.path.isfile(fcidvr_file1):
#        raise FileNotFoundError(f"N-1 electron file not found: {fcidvr_file1}")
#    if not os.path.isfile(fcidvr_file2):
#        raise FileNotFoundError(f"N electron file not found: {fcidvr_file2}")
#
#    base1 = fcidvr_file1.removesuffix(".npz"); base2 = fcidvr_file2.removesuffix(".npz")
#    parts1 = base1.split("_"); parts2 = base2.split("_")
#    nele1 = int(parts1[1][0]); nele2 = int(parts2[1][0])
#    spin1 = parts1[2]; spin2 = parts2[2]
#
#    assert nele2 - nele1 == 1, f"fcidvr_file2 must be the N electron file while fcidvr_file1 must be the N-1 electron file."
#    validate_nele_spin(nele1, spin1, Sz1); validate_nele_spin(nele2, spin2, Sz2)
#
#    fcidvr_data1 = np.load(fcidvr_file1); fcidvr_data2 = np.load(fcidvr_file2)
#    xi = fcidvr_data1['xi']
#    Cn1 = fcidvr_data1['Cn']; Cn2 = fcidvr_data2['Cn']
#    ndvr = xi.shape[0]
#    nbo1 = Cn1.shape[1]; nbo2 = Cn2.shape[1]
#    Dispn = np.zeros((ndvr, 2, nbo1, nbo2), dtype=np.complex128)
#
#    for p in range(nbo1):
#        print(p)
#        for n in range(nbo2):
#            wfn_p = fci_wfn(ndvr, nele1, spin1, Sz1, Cn1[:,p])
#            wfn_n = fci_wfn(ndvr, nele2, spin2, Sz2, Cn2[:,n])
#            Dispn[:,:,p,n] = np.sqrt(nele2)*(wfn_p.conj()[None,None,...] * wfn_n).sum(axis=tuple(range(2, int(2*nele2))))
#
#    np.savez(f'fcidvr_{nele1}ele_{spin1}_{Sz1}Sz_{nele2}ele_{spin2}_{Sz2}Sz_dyson', xi=xi, Dispn=Dispn)
#
#    return None




def solve_d1En(d1Hnm):
    return np.diag(d1Hnm).real

def solve_nac1(En, d1Hnm):
    nadi = En.shape[0]
    nac1 = np.zeros((nadi, nadi), dtype=np.complex128)
    mask = np.ones((nadi, nadi), dtype=bool)
    np.fill_diagonal(mask, False)
    DE = En[:, None] - En[None, :]
    nac1[mask] = -d1Hnm[mask] / DE[mask]
    return nac1

def solve_d2En(En, d1Hnm, d2Hnm):
    nadi = En.shape[0]
    d2En = np.zeros((nadi), dtype=np.float64)
    M = np.zeros((nadi, nadi), dtype=np.float64)
    mask = np.ones((nadi, nadi), dtype=bool)
    np.fill_diagonal(mask, False)
    DE = En[:, None] - En[None, :]
    M[mask] = np.absolute(d1Hnm[mask])**2 / DE[mask]
    d2En = np.diagonal(d2Hnm.real) + 2 * np.sum(M, axis=1)
    return d2En

def solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm):
    nadi = En.shape[0]
    nac2 = np.zeros((nadi, nadi), dtype=np.complex128)
    mask = np.ones((nadi, nadi), dtype=bool)
    np.fill_diagonal(mask, False)
    DE = En[:, None] - En[None, :]
    Dd1E = d1En[:, None] - d1En[None, :]
    nac2 += nac1 @ nac1
    nac2[mask] += ((nac1 @ d1Hnm - d1Hnm @ nac1) - Dd1E * nac1 - d2Hnm)[mask] / DE[mask]
    return nac2


#    def kernel(self, R: float, nele: int = 1, spin: str = 'doublet', nbo: int = 100):
#        self.validate_nele_spin(nele, spin)
#        xi = self.xi()
#        hij = self.hij(R)
#        Vik = self.Vik()
#        H = self.CIoperator(nele, spin, hij, Vik)
#        X = self.CIoperator(nele, spin, np.diag(xi))
#        d1H = self.CIoperator(nele, spin, self.d1hij(R))
#        d2H = self.CIoperator(nele, spin, self.d2hij(R))
#
#        En, Cn = eigsh(H, k = nbo, which = 'SA')
#        idx = np.argsort(En)
#        En = En[idx]; Cn = Cn[:,idx]
#        Cn *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,:])
#
#        En += self.model.VR(R)
#        xnm = Cn.conj().T @ X.matmat(Cn)
#        d1Hnm = Cn.conj().T @ d1H.matmat(Cn) + self.model.d1VR(R) * np.eye(nbo)
#        d2Hnm = Cn.conj().T @ d2H.matmat(Cn) + self.model.d2VR(R) * np.eye(nbo)
#
#        d1En = solve_d1En(d1Hnm)
#        nac1 = solve_nac1(En, d1Hnm)
#        d2En = solve_d2En(En, d1Hnm, d2Hnm)
#        nac2 = solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)
#
#        np.savez(f'fcidvr_{nele}ele_{spin}', xi=xi, En=En, Cn=Cn, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
#        return self



####################################################################################################################################################################

#    def HamiltonianOperator(self, R: float, nele, spin: str):
#        self.validate_nele_spin(nele, spin)
#        nfci, nmap, CImap = self.CImapping(nele, spin)
#        hij = self.hij(R)
#        Vik = self.Vik()
#        if nele == 1:
#            match spin:
#                case 'doublet':
#                    def matvec(C):
#                        return hij @ C
#        if nele == 2:
#            match spin:
#                case 'singlet':
#                    def matvec(C):
#                        Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
#                        Cjl[CImap] = C; Cjl *= nmap; Cjl += Cjl.T
#                        hC = hij @ Cjl
#                        Cik = nmap * (hC + hC.T + Vik * Cjl)
#                        return Cik[CImap]
#                case 'triplet':
#                    def matvec(C):
#                        Cjl = np.zeros((self.nxdvr, self.nxdvr), dtype=np.complex128)
#                        Cjl[CImap] = C; Cjl -= Cjl.T
#                        hC = hij @ Cjl
#                        Cik = (hC + hC.T + Vik * Cjl)
#                        return Cik[CImap]
#        def matmat(Cn):
#            n = Cn.shape[1]
#            res = np.zeros_like(Cn, dtype=np.complex128)
#            for j in range(n):
#                res[:, j] = matvec(Cn[:, j])
#            return res
#        return LinearOperator(shape=(nfci, nfci), matvec=matvec, matmat=matmat, dtype=np.complex128)

#    def kernel(self, R: float, nele: int = 1, spin: str = 'singlet', nbo: int = 100):
#        self.validate_nele_spin(nele, spin)
#
#        xi = self.xi()
#        En, Cn = eigsh(self.HamiltonianOperator(R), k=nbo, which='SA')
#        idx = np.argsort(En)
#        En = En[idx]; Cn = Cn[:,idx]; Cijn = self.vec_to_wfn(Cn)
#        Cijn *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,None,:])
#
#        xnm = Cn.conj().T @ self.CustomOperator(np.diag(xi)).matmat(Cn)
#        En += self.model.VR(R)
#        d1Hnm = Cn.conj().T @ self.CustomOperator(self.d1hij(R)).matmat(Cn) + self.model.d1VR(R) * np.eye(nbo)
#        d1En = np.diag(d1Hnm.real)
#        nac1 = self.solve_nac1(En, d1Hnm)
#        d2Hnm = Cn.conj().T @ self.CustomOperator(self.d2hij(R)).matmat(Cn) + self.model.d2VR(R) * np.eye(nbo)
#        d2En = self.solve_d2En(En, d1Hnm, d2Hnm)
#        nac2 = self.solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)
#
#        np.savez('2efci', xi=xi, En=En, Cijn=Cijn, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
#        return self

#    def solve_fci(self, R: float, nbo: int = 100):
#        xi = self.xi()
#        En, Cin = eigh(self.hij(R), subset_by_index = (0, nbo - 1))
#        Cin *= np.exp(1j*np.random.uniform(0, 2*np.pi, size=nbo)[None,:])
#
#        xnm = Cin.conj().T @ (xi[:,None] * Cin)
#        En += self.model.VR(R)
#        d1Hnm = Cin.conj().T @ self.d1hij(R) @ Cin + self.model.d1VR(R) * np.eye(nbo)
#        d1En = np.diag(d1Hnm.real)
#        nac1 = self.solve_nac1(En, d1Hnm)
#        d2Hnm = Cin.conj().T @ self.d2hij(R) @ Cin + self.model.d2VR(R) * np.eye(nbo)
#        d2En = self.solve_d2En(En, d1Hnm, d2Hnm)
#        nac2 = self.solve_nac2(En, d1Hnm, d1En, nac1, d2Hnm)
#
#        np.savez('1efci', xi=xi, En=En, Cin=Cin, xnm=xnm, d1Hnm=d1Hnm, d1En=d1En, nac1=nac1, d2Hnm=d2Hnm, d2En=d2En, nac2=nac2)
#        return self

