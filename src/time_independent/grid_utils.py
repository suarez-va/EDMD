import os
import numpy as np
from scipy.linalg import eigh, eig
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh, LinearOperator
from dvr_basis.colbert_miller_dvr import dvr_p, dvr_T

from model_systems.models import Model
from time_independent.fcidvr import fci_mapping, fci_operator, fci_wfn, FCIDVR

# For creating grid along R
def create_R_grid(Ra: float, Rb: float, RN: int, input_file: str):
    os.makedirs("RI", exist_ok=True)
    RI = np.linspace(Ra, Rb, RN+1)
    np.savetxt(os.path.join("RI", "RI.dat"), RI)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    with open(input_file, "r") as f:
        input_content = f.read()

    for I, R in enumerate(RI):
        sub_dir = os.path.join("RI", f"R{I}")
        os.makedirs(sub_dir, exist_ok=True)

        modified_content = input_content.replace("R_sub", f"{R:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(input_file)), "w") as f:
            f.write(modified_content)

    return None

def sort_R_grid(output_file: str):
    if not os.path.exists("RI"):
        print("Missing grid data directory RI")
        exit()
    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)

    for I, R in enumerate(RI):
        print(I)
        sub_dir_curr = f"RI/R{I}/"
        output_curr = np.load(sub_dir_curr + output_file)
        xi_curr = output_curr['xi']; xnm_curr = output_curr['xnm']; pnm_curr = output_curr['pnm']
        En_curr = output_curr['En']; Cn_curr = output_curr['Cn']
        d1En_curr = output_curr['d1En']; d1Cn_curr = output_curr['d1Cn']
        d2En_curr = output_curr['d2En']; d2Cn_curr = output_curr['d2Cn']
        nac01_curr = output_curr['nac01']; nac11_curr = output_curr['nac11']; nac02_curr = output_curr['nac02']
        nbo = En_curr.shape[0]

        if I == 0:
            theta = np.zeros((nbo), dtype=np.float64)
            for n in range(nbo):
                imax = np.argmax(np.abs(Cn_curr[:,n]))
                theta[n] = -np.angle(Cn_curr[imax,n])
        else:
            sub_dir_prev = f"RI/R{I-1}/"
            output_prev = np.load(sub_dir_prev + output_file)
            Cn_prev = output_prev['Cn']
            M_prev = np.abs(Cn_prev); A_prev = np.angle(Cn_prev)
            M_curr = np.abs(Cn_curr); A_curr = np.angle(Cn_curr)
            theta = -0.5 * np.sum((M_curr**2 + M_prev**2) * ((A_curr - A_prev + np.pi) % (2*np.pi) - np.pi), axis=0)

        Cn_curr = Cn_curr * np.exp(1j * theta[None,:])
        xnm_curr = xnm_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        pnm_curr = pnm_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        d1Cn_curr = d1Cn_curr * np.exp(1j * theta[None,:])
        d2Cn_curr = d2Cn_curr * np.exp(1j * theta[None,:])
        nac01_curr = nac01_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        nac11_curr = nac11_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        nac02_curr = nac02_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))

        np.savez(sub_dir_curr + output_file, xi=xi_curr, pnm=pnm_curr, xnm=xnm_curr, En=En_curr, Cn=Cn_curr, d1En=d1En_curr, d1Cn=d1Cn_curr, d2En=d2En_curr, d2Cn=d2Cn_curr, nac01=nac01_curr, nac11=nac11_curr, nac02=nac02_curr)

    return None

def save_R_grid(output_file: str, nbo: int):
    if not os.path.exists("RI"):
        print("Missing grid data directory RI")
        exit()
    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)
    Rpts = RI.shape[0]
    EnI = np.zeros((nbo, Rpts), dtype=np.float64)
    d1EnI = np.zeros((nbo, Rpts), dtype=np.float64)
    nac01I = np.zeros((nbo, nbo, Rpts), dtype=np.complex128)
    nac11I = np.zeros((nbo, nbo, Rpts), dtype=np.complex128)

    for I, R in enumerate(RI):
        print(I)
        sub_dir_curr = f"RI/R{I}/"
        output_curr = np.load(sub_dir_curr + output_file)
        En_curr = output_curr['En']; d1En_curr = output_curr['d1En']
        nac01_curr = output_curr['nac01']; nac11_curr = output_curr['nac11']

        EnI[:,I] = En_curr[:nbo]
        d1EnI[:,I] = d1En_curr[:nbo]
        nac01I[:,:,I] = nac01_curr[:nbo,:nbo]
        nac11I[:,:,I] = nac11_curr[:nbo,:nbo]

    np.savez('RI/' + output_file[:-4] + '_nucgrid.npz', RI=RI, EnI=EnI, d1EnI=d1EnI, nac01I=nac01I, nac11I=nac11I)

    return None

def MolecularHamiltonian(nucgrid_file: str, mass: float = 1836.0):
    if not os.path.isfile(nucgrid_file):
        raise FileNotFoundError(f"File not found: {nucgrid_file}")

    if not nucgrid_file.endswith(".npz"):
        raise ValueError(f"File must be a .npz file: {nucgrid_file}")

    with np.load(nucgrid_file) as data:
        required_keys = ['RI', 'EnI', 'd1EnI', 'nac01I', 'nac11I']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise KeyError(f"Missing required keys in .npz file: {missing_keys}")

    data = np.load(nucgrid_file)
    nbo = data['EnI'].shape[0]
    Ra = data['RI'][0]
    Rb = data['RI'][-1]
    RN = data['RI'].shape[0] - 1
    Rbounds = '(-inf,inf)'
    nRdvr = RN + 1
    nexact = nbo * nRdvr
    pIJ = dvr_p(Ra, Rb, RN, Rbounds)
    TIJ = dvr_T(mass, Ra, Rb, RN, Rbounds)

    nacI01 = data['nac01I'].transpose(2,0,1)
    nacI11 = data['nac11I'].transpose(2,0,1)

    def matvec(C):
        CmJ = C.reshape(nbo, nRdvr)
        dCnJ = (nacI01 @ CmJ.T[:,:,None]).squeeze(-1).T
        pCmI = (pIJ @ CmJ.T).T
        CnI = ((TIJ @ CmJ.T).T
            - 0.5j/mass*(pIJ @ dCnJ.T).T
            - 0.5j/mass*(nacI01 @ pCmI.T[:,:,None]).squeeze(-1).T
            + 0.5/mass*(nacI11 @ CmJ.T[:,:,None]).squeeze(-1).T
            + data['EnI']*CmJ)
        return CnI.reshape(nexact)

    def matmat(CM):
        M = CM.shape[1]
        CN = np.zeros_like(CM, dtype=np.complex128)
        for j in range(M):
            CN[:, j] = matvec(CM[:, j])
        return CN

    return LinearOperator(shape=(nexact, nexact), matvec=matvec, matmat=matmat, dtype=np.complex128)

def save_R_grid_cap(output_file: str, nbo: int):
    if not os.path.exists("RI"):
        print("Missing grid data directory RI")
        exit()
    base = output_file.removesuffix(".npz")
    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)
    Rpts = RI.shape[0]
    wi = np.load('RI/R0/' + base + '_cap.npz')['wi']
    WnmI = np.zeros((nbo, nbo, Rpts), dtype=np.complex128)

    for I, R in enumerate(RI):
        print(I)
        sub_dir_curr = f"RI/R{I}/"
        output_curr = np.load(sub_dir_curr + base + '_cap.npz')
        Wnm_curr = output_curr['Wnm']

        WnmI[:,:,I] = Wnm_curr[:nbo,:nbo]

    np.savez('RI/' + base + '_nucgrid_cap.npz', wi=wi, RI=RI, WnmI=WnmI)

    return None


def solve_cap(Hnm: np.ndarray, Wnm: np.ndarray, eta: float, output_file: str = 'cap'):
    Hcap = Hnm - 1j * eta *  Wnm
    En, Cnml, Cnmr = eig(Hcap, left=True, right=True)
    idx = np.argsort(En.real)
    En, Cnml, Cnmr = En[idx], Cnml[:,idx], Cnmr[:,idx]
    norm = np.sqrt(np.diag(np.matmul(Cnml.conj().T, Cnmr)))
    Cnml *= 1 / norm.conj(); Cnmr *= 1 / norm
    np.savez(output_file, En=En, Cnml=Cnml, Cnmr=Cnmr)
    return En, Cnml, Cnmr

def create_eta_grid(coef, delta, ipts, template_file):
    os.makedirs("etai", exist_ok=True)
    etai = coef * (delta**np.arange(ipts+1)-1) / (delta - 1)
    np.savetxt(os.path.join("etai", "etai.dat"), etai)

    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' not found.")
        return

    with open(template_file, "r") as f:
        template_content = f.read()

    for i, eta in enumerate(etai):
        sub_dir = os.path.join("etai", f"eta{i}")
        os.makedirs(sub_dir, exist_ok=True)

        modified_content = template_content.replace("eta_sub", f"{eta:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(template_file)), "w") as f:
            f.write(modified_content)

    return None

def sort_eta_grid(output_file: str = 'cap'):
    if not os.path.exists("etai"):
        print("Missing grid data directory etai")
        exit()
    etai = np.loadtxt("etai/etai.dat", dtype=np.float64)

    for i, eta in enumerate(etai[1:], start=1):
        sub_dir_curr = f"etai/eta{i}/"
        sub_dir_prev = f"etai/eta{i-1}/"
        capspec_curr = np.load(sub_dir_curr + output_file)
        capspec_prev = np.load(sub_dir_prev + output_file)
        En_curr = capspec_curr["En"] ; Cnml_curr = capspec_curr["Cnml"]; Cnmr_curr = capspec_curr["Cnmr"]
        Cnml_prev = capspec_prev["Cnml"]; Cnmr_prev = capspec_prev["Cnmr"]

        ovlp = np.matmul(Cnml_prev.conj().T, Cnmr_curr)
        row_ind, col_ind = linear_sum_assignment(-np.absolute(ovlp)**2)
        np.savez(sub_dir_curr + output_file, En=En_curr[col_ind], Cnml=Cnml_curr[:,col_ind], Cnmr=Cnmr_curr[:,col_ind])

    return None

def save_eta_grid(output_file: str = 'cap'):
    if not os.path.exists("etai"):
        print("Missing grid data directory etai")
        exit()
    etai = np.loadtxt("etai/etai.dat", dtype=np.float64)
    nbo = np.load('etai/eta0/' + output_file)['En'].shape[0]
    ipts = etai.shape[0]
    Eni = np.zeros((nbo, ipts), dtype=np.complex128)

    for i, eta in enumerate(etai):
        print(i)
        sub_dir_curr = f"etai/eta{i}/"
        Eni[:,i] = np.load(sub_dir_curr + output_file)['En']

    np.savez('etai/' + output_file[:-4] + '_capgrid.npz', etai=etai, Eni=Eni)

    return None


# For creating grid along L
def create_L_grid(La: float, Lb: float, LN: int, input_file: str):
    os.makedirs("Ln", exist_ok=True)
    Ln = np.linspace(La, Lb, LN+1)
    np.savetxt(os.path.join("Ln", "Ln.dat"), Ln)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    with open(input_file, "r") as f:
        input_content = f.read()

    for n, L in enumerate(Ln):
        sub_dir = os.path.join("Ln", f"L{n}")
        os.makedirs(sub_dir, exist_ok=True)

        modified_content = input_content.replace("L_sub", f"{L:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(input_file)), "w") as f:
            f.write(modified_content)

    return None



class NUCDVR:
    def __init__(self, nucgrid_file: str, mass: float = 1836.0):
        self.nucgrid_file = nucgrid_file
        self.mass = mass

        if not os.path.isfile(self.nucgrid_file):
            raise FileNotFoundError(f"File not found: {self.nucgrid_file}")

        if not self.nucgrid_file.endswith(".npz"):
            raise ValueError(f"File must be a .npz file: {self.nucgrid_file}")

        with np.load(self.nucgrid_file) as data:
            required_keys = ['RI', 'EnI', 'd1EnI', 'nac01I', 'nac11I']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                raise KeyError(f"Missing required keys in .npz file: {missing_keys}")

        self.data = np.load(self.nucgrid_file)
        self.Ra = self.data['RI'][0]
        self.Rb = self.data['RI'][-1]
        self.RN = self.data['RI'].shape[0] - 1
        self.Rbounds = '(-inf,inf)'
        self.nRdvr = self.RN + 1

        self.nbo = self.data['EnI'].shape[0]
        self.nfci = self.nbo * self.nRdvr

    def RI(self) -> np.ndarray:
        return np.linspace(self.Ra, self.Rb, self.nRdvr)

    def pIJ(self) -> np.ndarray:
        return dvr_p(self.Ra, self.Rb, self.RN, self.Rbounds)

    def TIJ(self) -> np.ndarray:
        return dvr_T(self.mass, self.Ra, self.Rb, self.RN, self.Rbounds)

    def HamiltonianOperator(self):
        pIJ = self.pIJ()
        TIJ = self.TIJ()

        def matvec(C):
            CmJ = C.reshape(self.nbo, self.nRdvr)
            dCnJ = np.einsum('nmJ,mJ->nJ', self.data['nac01I'], CmJ)
            pCmI = np.einsum('IJ,mJ->mI', pIJ, CmJ)
            CnI = np.einsum('IJ,nJ->nI', TIJ, CmJ) - 0.5j/self.mass*np.einsum('IJ,nJ->nI', pIJ, dCnJ) - 0.5j/self.mass*np.einsum('nmI,mI->nI', self.data['nac01I'], pCmI) + 0.5/self.mass*np.einsum('nmI,mI->nI', self.data['nac11I'], CmJ) + self.data['EnI']*CmJ
            return CnI.reshape(self.nfci)

        def matmat(CM):
            M = CM.shape[1]
            CN = np.zeros_like(CM, dtype=np.complex128)
            for j in range(M):
                CN[:, j] = matvec(CM[:, j])
            return CN

        return LinearOperator(shape=(self.nfci, self.nfci), matvec=matvec, matmat=matmat, dtype=np.complex128)

class FCINUCDVR(FCIDVR):
    def __init__(self, model: Model, xa: float, xb: float, xN: int, xbounds: str, mass: float, Ra: float, Rb: float, RN: int):
        super().__init__(model, xa, xb, xN, xbounds)

        self.mass = mass
        self.Ra = Ra
        self.Rb = Rb
        self.RN = RN
        self.Rbounds = '(-inf,inf)'
        self.nRdvr = self.RN + 1

    def RI(self) -> np.ndarray:
        return np.linspace(self.Ra, self.Rb, self.nRdvr)

    def TIJ(self) -> np.ndarray:
        return dvr_T(self.mass, self.Ra, self.Rb, self.RN, self.Rbounds)

    def H_NUC(self, nele: int, spin: str):
        #self.validate_nele_spin(nele, spin)
        nfci, nmap, CImap = fci_mapping(self.nxdvr, nele, spin)
        RI = self.RI()
        TIJ = self.TIJ()
        hijI = np.zeros((self.nxdvr, self.nxdvr, self.nRdvr), dtype=np.complex128)
        gik = self.gik()
        VRI = np.zeros((self.nRdvr))
        for I, R in enumerate(RI):
            hijI[:,:,I] = self.hij(R)
            VRI[I] = self.model.VR(R)

        matvec = lambda C: C
        if nele == 1:
            match spin:
                case 'doublet':
                    def matvec(C):
                        CjJ = C.reshape(nfci, self.nRdvr)
                        CiI = np.einsum('IJ,iJ->iI', TIJ, CjJ) + np.einsum('ijI,jI->iI', hijI, CjJ) + VRI[None,:]*CjJ
                        return CiI.reshape(nfci*self.nRdvr)
        if nele == 2:
            match spin:
                case 'singlet':
                    def matvec(C):
                        return C
                case 'triplet':
                    def matvec(C):
                        return C
        if nele == 3:
            match spin:
                case 'doublet':
                    def matvec(C):
                        return C

        def matmat(Cm):
            m = Cm.shape[1]
            Cn = np.zeros_like(Cm, dtype=np.complex128)
            for j in range(m):
                Cn[:, j] = matvec(Cm[:, j])
            return Cn
        return LinearOperator(shape=(nfci*self.nRdvr, nfci*self.nRdvr), matvec=matvec, matmat=matmat, dtype=np.complex128)


#def sort_R_grid(output_file: str):
#    if not os.path.exists("RI"):
#        print("Missing grid data directory RI")
#        exit()
#    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)
#
#    for I, R in enumerate(RI):
#        print(I)
#        sub_dir_curr = f"RI/R{I}/"
#        output_curr = np.load(sub_dir_curr + output_file)
#        xi_curr = output_curr['xi']; En_curr = output_curr['En']; Cn_curr = output_curr['Cn']
#        xnm_curr = output_curr['xnm']#; pnm_curr = output_curr['pnm']; Tnm_curr = output_curr['Tnm']
#        d1Hnm_curr = output_curr['d1Hnm']; d1En_curr = output_curr['d1En']; nac1_curr = output_curr['nac1']
#        d2Hnm_curr = output_curr['d2Hnm']; d2En_curr = output_curr['d2En']; nac2_curr = output_curr['nac2']
#        nbo = En_curr.shape[0]
#
#        if I == 0:
#            theta = np.zeros((nbo), dtype=np.float64)
#            for n in range(nbo):
#                imax = np.argmax(np.abs(Cn_curr[:,n]))
#                theta[n] = -np.angle(Cn_curr[imax,n])
#        else:
#            sub_dir_prev = f"RI/R{I-1}/"
#            output_prev = np.load(sub_dir_prev + output_file)
#            Cn_prev = output_prev['Cn']
#            M_prev = np.abs(Cn_prev); A_prev = np.angle(Cn_prev)
#            M_curr = np.abs(Cn_curr); A_curr = np.angle(Cn_curr)
#            theta = -0.5 * np.sum((M_curr**2 + M_prev**2) * ((A_curr - A_prev + np.pi) % (2*np.pi) - np.pi), axis=0)
#
#        Cn_curr = Cn_curr * np.exp(1j * theta[None,:])
#        xnm_curr = xnm_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
#        d1Hnm_curr = d1Hnm_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
#        nac1_curr = nac1_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
#        d2Hnm_curr = d2Hnm_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
#        nac2_curr = nac2_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
#
#        np.savez(sub_dir_curr + output_file, xi=xi_curr, En=En_curr, Cn=Cn_curr, xnm=xnm_curr, d1Hnm=d1Hnm_curr, d1En=d1En_curr, nac1=nac1_curr, d2Hnm=d2Hnm_curr, d2En=d2En_curr, nac2=nac2_curr)
#
#    return None

#def save_R_grid(output_file: str, nbo: int):
#    if not os.path.exists("RI"):
#        print("Missing grid data directory RI")
#        exit()
#    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)
#    Rpts = RI.shape[0]
#    EnI = np.zeros((nbo, Rpts), dtype=np.float64)
#    d1EnI = np.zeros((nbo, Rpts), dtype=np.float64)
#    nac1I = np.zeros((nbo, nbo, Rpts), dtype=np.complex128)
#    nac2I = np.zeros((nbo, nbo, Rpts), dtype=np.complex128)
#
#    for I, R in enumerate(RI):
#        print(I)
#        sub_dir_curr = f"RI/R{I}/"
#        output_curr = np.load(sub_dir_curr + output_file)
#        En_curr = output_curr['En']
#        d1En_curr = output_curr['d1En']; nac1_curr = output_curr['nac1']
#        nac2_curr = output_curr['nac2']
#
#        EnI[:,I] = En_curr[:nbo]
#        d1EnI[:,I] = d1En_curr[:nbo]
#        nac1I[:,:,I] = nac1_curr[:nbo,:nbo]
#        nac2I[:,:,I] = nac2_curr[:nbo,:nbo]
#
#    np.savez('RI/' + output_file[:-4] + '_nucgrid.npz', RI=RI, EnI=EnI, d1EnI=d1EnI, nac1I=nac1I, nac2I=nac2I)
#
#    return None


