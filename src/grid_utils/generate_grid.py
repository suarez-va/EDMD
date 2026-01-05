import os
import numpy as np

def create_directories(Ra, Rb, RN, template_file):
    os.makedirs("Rk", exist_ok=True)
    np.savetxt(os.path.join("Rk", "Rgrid.dat"), np.linspace(Ra, Rb, RN+1)[1:-1])
        
    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' not found.")
        return

    with open(template_file, "r") as f:
        template_content = f.read()

    for i in range(1, RN):
        sub_dir = os.path.join("Rk", f"R{i}")
        os.makedirs(sub_dir, exist_ok=True)

        Ri = Ra + i * (Rb - Ra) / RN
        modified_content = template_content.replace("REPLACE", f"{Ri:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(template_file)), "w") as f:
            f.write(modified_content)

    return None


# The change you want to make is here, make this return a dictionary with all your grid data
def retrieve_data(nstates = 2, datalist = ['Eadi', 'nac1', 'nac2']):
    if not os.path.exists("Rk"):
        print("Missing grid data directory Rk")
        exit()
    Rpts = int(os.popen("wc -l < Rk/Rgrid.dat").read().strip())

#    Eadi = np.zeros((Rpts, nstates))
#    nac1 = np.zeros((Rpts, nstates**2), dtype=complex)
#    nac2 = np.zeros((Rpts, nstates**2), dtype=complex)
#    dipole = np.zeros((Rpts, nstates**2), dtype=complex)
#    cap = np.zeros((Rpts, nstates**2), dtype=complex)

    for i in range(1, Rpts + 1):
        sub_dir = os.path.join("Rk", f"R{i}")
        Eadi[i-1,:] = np.loadtxt(os.path.join(sub_dir, "E.dat"))[:nstates]
        nac1[i-1,:] = np.loadtxt(os.path.join(sub_dir, "nac1.dat",), dtype=complex)[:nstates,:nstates].reshape(nstates**2)
        nac2[i-1,:] = np.loadtxt(os.path.join(sub_dir, "nac2.dat",), dtype=complex)[:nstates,:nstates].reshape(nstates**2)
        dipole[i-1,:] = np.loadtxt(os.path.join(sub_dir, "dipole.dat",), dtype=complex)[:nstates,:nstates].reshape(nstates**2)
        cap[i-1,:] = np.loadtxt(os.path.join(sub_dir, "cap.dat",), dtype=complex)[:nstates,:nstates].reshape(nstates**2)

    np.savetxt(os.path.join("Rk", "Eadi.dat"), Eadi)
    np.savetxt(os.path.join("Rk", "nac1.dat"), nac1)
    np.savetxt(os.path.join("Rk", "nac2.dat"), nac2)
    np.savetxt(os.path.join("Rk", "dipole.dat"), dipole)
    np.savetxt(os.path.join("Rk", "cap.dat"), cap)

    return None


def create_eta_grid(coef, delta, npts, template_file):
    eta_ar = coef * (delta**np.arange(npts+1)-1) / (delta - 1)
    os.makedirs("etan", exist_ok=True)
    np.savetxt(os.path.join("etan", "eta.dat"), eta_ar)
        
    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' not found.")
        return

    with open(template_file, "r") as f:
        template_content = f.read()

    for n in range(npts+1):
        sub_dir = os.path.join("etan", f"eta{n}")
        os.makedirs(sub_dir, exist_ok=True)

        modified_content = template_content.replace("REPLACE", f"{eta_ar[n]:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(template_file)), "w") as f:
            f.write(modified_content)

    return None

def sort_eta_data():
    if not os.path.exists("etan"):
        print("Missing grid data directory etan")
        exit()
    npts = int(os.popen("wc -l < etan/eta.dat").read().strip()) - 1

    for n in range(1, npts + 1):
        sub_dir_prev = os.path.join("etan", f"eta{n-1}")
        sub_dir_curr = os.path.join("etan", f"eta{n}")
        Clnm_prev = np.loadtxt(os.path.join(sub_dir_prev, "Clnm.dat"), dtype=np.complex128)
        Crnm_prev = np.loadtxt(os.path.join(sub_dir_prev, "Crnm.dat"), dtype=np.complex128)
        Er_curr= np.loadtxt(os.path.join(sub_dir_curr, "Er.dat"))
        Gam_curr = np.loadtxt(os.path.join(sub_dir_curr, "Gam.dat"))
        Clnm_curr = np.loadtxt(os.path.join(sub_dir_curr, "Clnm.dat"), dtype=np.complex128)
        Crnm_curr = np.loadtxt(os.path.join(sub_dir_curr, "Crnm.dat"), dtype=np.complex128)
        #ovlp = np.einsum("ijn,ijm->nm", Cijn_prev, Cijn_curr)
        #ovlp = np.absolute(np.tensordot(Cijn_prev, Cijn_curr, axes=([0,1],[0,1])))
        #ovlp = np.matmul(Cnm_prev.conj().T, Cnm_curr)
        ovlp = np.matmul(Clnm_prev.conj().T, Crnm_curr)
        #idx = np.argmax(np.absolute(ovlp), axis=1)
        idx = fix_index_array(np.argmax(np.absolute(ovlp), axis=1))[0]
        Er_sort = Er_curr[idx]
        Gam_sort = Gam_curr[idx]
        Clnm_sort = Clnm_curr[:,idx]
        Crnm_sort = Crnm_curr[:,idx]
        np.savetxt(os.path.join(sub_dir_curr, "Er.dat"), Er_sort)
        np.savetxt(os.path.join(sub_dir_curr, "Gam.dat"), Gam_sort)
        np.savetxt(os.path.join(sub_dir_curr, "Clnm.dat"), Clnm_sort)
        np.savetxt(os.path.join(sub_dir_curr, "Crnm.dat"), Crnm_sort)
        print(idx)
        if idx.size != np.unique(idx).size:
            print('DUPLICATE ABOVE!!!')
            vals, counts = np.unique(idx, return_counts=True)
            print(vals[counts > 1])

    return None

def fix_index_array(arr):
    """
    AI Generated:
    Fix an integer array so that each value in the range [0, max(arr)]
    appears exactly once. Later duplicates are replaced by the lowest missing elements.

    Parameters 
    ---------- 
    arr : np.ndarray
        1D array of integers representing index mappings.
    
    Returns
    -------
    arr_fixed : np.ndarray
        Array with duplicates replaced by missing elements.
    missing : np.ndarray
        The missing elements that were used to fix duplicates.
    duplicate_values : np.ndarray
        Values that had duplicates in the original array.
    """
    arr = np.asarray(arr)
    expected = np.arange(arr.max() + 1)
    missing = np.setdiff1d(expected, arr)
    unique_vals, first_idx, counts = np.unique(arr, return_index=True, return_counts=True)
    duplicate_values = unique_vals[counts > 1]
    mask_later_duplicates = np.zeros(len(arr), dtype=bool)
    for val in duplicate_values:
        indices = np.where(arr == val)[0]
        mask_later_duplicates[indices[1:]] = True
    arr_fixed = arr.copy()
    arr_fixed[mask_later_duplicates] = missing
    return arr_fixed, missing, duplicate_values


def get_eta_data(idx):
    if not os.path.exists("etan"):
        print("Missing grid data directory etan")
        exit()
    npts = int(os.popen("wc -l < etan/eta.dat").read().strip()) - 1

    eta_data = np.loadtxt("etan/eta.dat")
    Er_data = np.zeros(eta_data.shape[0])
    Gam_data = np.zeros(eta_data.shape[0])

    for n in range(npts + 1):
        sub_dir_curr = os.path.join("etan", f"eta{n}")
        Er_curr= np.loadtxt(os.path.join(sub_dir_curr, "Er.dat"))
        Gam_curr = np.loadtxt(os.path.join(sub_dir_curr, "Gam.dat"))
        Er_data[n] = Er_curr[idx]
        Gam_data[n] = Gam_curr[idx]

    return eta_data, Er_data, Gam_data




def sort_eta_data_old():
    if not os.path.exists("etan"):
        print("Missing grid data directory etan")
        exit()
    npts = int(os.popen("wc -l < etan/eta.dat").read().strip()) - 1

    for n in range(1, npts + 1):
        sub_dir_prev = os.path.join("etan", f"eta{n-1}")
        sub_dir_curr = os.path.join("etan", f"eta{n}")
        #Er_prev = np.loadtxt(os.path.join(sub_dir_prev, "Er.dat"))
        #Gam_prev = np.loadtxt(os.path.join(sub_dir_prev, "Gam.dat"))
        Cijn_prev = np.load(os.path.join(sub_dir_prev, "Cijn.npy"))
        Er_curr= np.loadtxt(os.path.join(sub_dir_curr, "Er.dat"))
        Gam_curr = np.loadtxt(os.path.join(sub_dir_curr, "Gam.dat"))
        Cijn_curr = np.load(os.path.join(sub_dir_curr, "Cijn.npy"))
        #ovlp = np.einsum("ijn,ijm->nm", Cijn_prev, Cijn_curr)
        ovlp = np.absolute(np.tensordot(Cijn_prev, Cijn_curr, axes=([0,1],[0,1])))
        idx = np.argmax(ovlp, axis=1)
        Er_sort = Er_curr[idx]
        Gam_sort = Gam_curr[idx]
        Cijn_sort = Cijn_curr[:,:,idx]
        np.savetxt(os.path.join(sub_dir_curr, "Er.dat"), Er_sort)
        np.savetxt(os.path.join(sub_dir_curr, "Gam.dat"), Gam_sort)
        np.save(os.path.join(sub_dir_curr, "Cijn"), Cijn_sort)
        print(idx)

    #np.savetxt(os.path.join("Rk", "Eadi.dat"), Eadi)
    #np.savetxt(os.path.join("Rk", "nac1.dat"), nac1)
    #np.savetxt(os.path.join("Rk", "nac2.dat"), nac2)
    #np.savetxt(os.path.join("Rk", "dipole.dat"), dipole)
    #np.savetxt(os.path.join("Rk", "cap.dat"), cap)

    return None
