import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def create_R_grid(Ra, Rb, RN, template_file):
    os.makedirs("RI", exist_ok=True)
    RI = np.linspace(Ra, Rb, RN+1)
    np.savetxt(os.path.join("RI", "RI.dat"), RI)

    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' not found.")
        return

    with open(template_file, "r") as f:
        template_content = f.read()

    for I, R in enumerate(RI):
        sub_dir = os.path.join("RI", f"R{I}")
        os.makedirs(sub_dir, exist_ok=True)

        #R = Ra + k * (Rb - Ra) / RN
        modified_content = template_content.replace("R_sub", f"{R:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(template_file)), "w") as f:
            f.write(modified_content)

    return None

def sort_R_grid_mos():
    if not os.path.exists("RI"):
        print("Missing grid data directory RI")
        exit()
    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)

    for I, R in enumerate(RI):
        print(I)
        sub_dir_curr = f"RI/R{I}"
        mospec_curr = np.load(sub_dir_curr + "/mospec.npz")
        xi_curr = mospec_curr['xi']; ep_curr = mospec_curr['ep']; cip_curr = mospec_curr['cip']
        xpq_curr = mospec_curr['xpq']#; ppq_curr = mospec_curr['ppq']; Ppq_curr = mospec_curr['Tpq']
        d1hpq_curr = mospec_curr['d1hpq']; d1ep_curr = mospec_curr['d1ep']; nac1_curr = mospec_curr['nac1']
        d2hpq_curr = mospec_curr['d2hpq']; d2ep_curr = mospec_curr['d2ep']; nac2_curr = mospec_curr['nac2']
        nmo = ep_curr.shape[0]

        if I == 0:
            theta = np.zeros((nmo), dtype=np.float64)
            for p in range(nmo):
                imax = np.argmax(np.abs(cip_curr[:,p]))
                theta[p] = -np.angle(cip_curr[imax,p])
        else:
            sub_dir_prev = f"RI/R{I-1}"
            mospec_prev = np.load(sub_dir_prev + "/mospec.npz")
            cip_prev = mospec_prev['cip']
            M_prev = np.abs(cip_prev); A_prev = np.angle(cip_prev)
            M_curr = np.abs(cip_curr); A_curr = np.angle(cip_curr)
            theta = -0.5 * np.sum((M_curr**2 + M_prev**2) * ((A_curr - A_prev + np.pi) % (2*np.pi) - np.pi), axis=0)

        cip_curr = cip_curr * np.exp(1j * theta[None,:])
        xpq_curr = xpq_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        d1hpq_curr = d1hpq_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        nac1_curr = nac1_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        d2hpq_curr = d2hpq_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        nac2_curr = nac2_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))

        np.savez(sub_dir_curr + '/mospec', xi=xi_curr, ep=ep_curr, cip=cip_curr, xpq=xpq_curr, d1hpq=d1hpq_curr, d1ep=d1ep_curr, nac1=nac1_curr, d2hpq=d2hpq_curr, d2ep=d2ep_curr, nac2=nac2_curr)

    return None


def sort_R_grid_mos_old():
    if not os.path.exists("RI"):
        print("Missing grid data directory RI")
        exit()
    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)

    for I, R in enumerate(RI):
        print(I)
        sub_dir_curr = f"RI/R{I}"
        mospec_curr = np.load(sub_dir_curr + "/mospec.npz")
        xi_curr = mospec_curr['xi']; ep_curr = mospec_curr['ep']; cip_curr = mospec_curr['cip']
        xpq_curr = mospec_curr['xpq']#; ppq_curr = mospec_curr['ppq']; Ppq_curr = mospec_curr['Tpq']
        d1hpq_curr = mospec_curr['d1hpq']; d1ep_curr = mospec_curr['d1ep']; nac1_curr = mospec_curr['nac1']
        d2hpq_curr = mospec_curr['d2hpq']; d2ep_curr = mospec_curr['d2ep']; nac2_curr = mospec_curr['nac2']
        nmo = ep_curr.shape[0]

        if I == 0:
            theta = np.zeros((nmo), dtype=np.float64)
            #ep_old = 0
            #d1ep_old = 0
            #cip_prev = cip_curr
            #cip_old = cip_curr
            #nac1_prev = nac1_curr
            #nac1_old = nac1_curr
            #nac2_prev = nac2_curr
            for p in range(nmo):
                imax = np.argmax(np.abs(cip_curr[:,p]))
                theta[p] = -np.angle(cip_curr[imax,p])
        else:
            sub_dir_prev = f"RI/R{I-1}"
            mospec_prev = np.load(sub_dir_prev + "/mospec.npz")
            cip_prev = mospec_prev['cip']
            #nac1_prev = mospec_prev['nac1']
            M_prev = np.abs(cip_prev); A_prev = np.angle(cip_prev)
            M_curr = np.abs(cip_curr); A_curr = np.angle(cip_curr)
            theta = -0.5 * np.sum((M_curr**2 + M_prev**2) * ((A_curr - A_prev + np.pi) % (2*np.pi) - np.pi), axis=0)
            
            #nsta = 1
            #DR = RI[1] - RI[0]
            #ep_prev = mospec_prev['ep']
            #d1ep_prev = mospec_prev['d1ep']
            #d2ep_prev = mospec_prev['d2ep']
            #d1ep_approx1 = ((ep_curr - ep_prev) / DR)
            #d1ep_approx2 = ((ep_prev - ep_old) / DR)
            #d2ep_approx0 = ((ep_curr - 2*ep_prev + ep_old) / (DR**2))
            #d2ep_approx1 = ((d1ep_curr - d1ep_prev) / DR)
            #d2ep_approx2 = ((d1ep_prev - d1ep_old) / DR)
            ##print(f'1: {d1ep_approx1[nsta]}, 2: {d1ep_approx2[nsta]}, res: {d1ep_prev[nsta]}')
            ##print(f'0: {d2ep_approx0[nsta]}, 1: {d2ep_approx1[nsta]}, 2: {d2ep_approx2[nsta]}, res: {d2ep_prev[nsta]}')
            #ep_old = ep_prev
            #d1ep_old = d1ep_prev
            #nac1_prev = mospec_prev['nac1']
            #nac2_prev = mospec_prev['nac2']

        cip_curr = cip_curr * np.exp(1j * theta[None,:])
        xpq_curr = xpq_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        d1hpq_curr = d1hpq_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        nac1_curr = nac1_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        d2hpq_curr = d2hpq_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))
        nac2_curr = nac2_curr * np.exp(-1j * (theta[:,None] - theta[None,:]))

        #DR = RI[1] - RI[0]
        #print(f'0: {(cip_prev.conj().T @ cip_curr)[22,23]/DR}, res: {nac1_prev[22,23]}')
        #print(f'0: {(cip_prev.conj().T @ cip_curr + cip_prev.conj().T @ cip_old)[22,23]/(DR**2)}, res: {nac2_prev[22,23]}')
        #cip_old=cip_prev
        np.savez(sub_dir_curr + '/mospec', xi=xi_curr, ep=ep_curr, cip=cip_curr, xpq=xpq_curr, d1hpq=d1hpq_curr, d1ep=d1ep_curr, nac1=nac1_curr, d2hpq=d2hpq_curr, d2ep=d2ep_curr, nac2=nac2_curr)

    return None

def sort_R_grid_old():
    if not os.path.exists("RI"):
        print("Missing grid data directory RI")
        exit()
    RI = np.loadtxt("RI/RI.dat", dtype=np.float64)

    eigspec_0 = np.load("RI/R0/eigspec.npz")
    xi_0 = eigspec_0['xi']; En_0 = eigspec_0['En']; Cijn_0 = eigspec_0['Cijn']
    d1En_0 = eigspec_0['d1En']; d1Hnm_0 = eigspec_0['d1Hnm']; nac1_0 = eigspec_0['nac1']
    d2En_0 = eigspec_0['d2En']; d2Hnm_0 = eigspec_0['d2Hnm']
    ndvr = Cijn_0.shape[0]; nbo = Cijn_0.shape[2]

    Tn_0 = np.zeros((nbo), dtype=np.float64)
    for n in range(nbo):
        (imax, jmax) = np.unravel_index(np.argmax(np.absolute(Cijn_0[:,:,n])), (ndvr, ndvr))
        Tn_0[n] = -np.angle(Cijn_0[imax,jmax,n])
    Cijn_0 = Cijn_0 * np.exp(1j * Tn_0[None,None,:])
    d1Hnm_0 = d1Hnm_0 * np.exp(-1j * (Tn_0[:,None] - Tn_0[None,:]))
    nac1_0 = nac1_0 * np.exp(-1j * (Tn_0[:,None] - Tn_0[None,:]))
    d2Hnm_0 = d2Hnm_0 * np.exp(-1j * (Tn_0[:,None] - Tn_0[None,:]))
    np.savez("RI/R0/eigspec2", xi=xi_0, En=En_0, Cijn=Cijn_0, d1En=d1En_0, d1Hnm=d1Hnm_0, nac1=nac1_0, d2En=d2En_0, d2Hnm=d2Hnm_0)
    if os.path.isfile("RI/R0/boops.npz"):
        boops = {}
        boops_0 = np.load("RI/R0/boops.npz")
        for key in boops_0.files:
            boops[key] = boops_0[key] * np.exp(-1j * (Tn_0[:,None] - Tn_0[None,:]))
        np.savez("RI/R0/boops2", **boops)

    for k, R in enumerate(RI[1:], start=1):
        print(k)
        sub_dir_prev = f"RI/R{k-1}"
        sub_dir_curr = f"RI/R{k}"
        eigspec_prev = np.load(sub_dir_prev + "/eigspec2.npz")
        eigspec_curr = np.load(sub_dir_curr + "/eigspec.npz")
        Cijn_prev = eigspec_prev['Cijn']
        xi_curr = eigspec_curr['xi']; En_curr = eigspec_curr['En']; Cijn_curr = eigspec_curr['Cijn']
        d1En_curr = eigspec_curr['d1En']; d1Hnm_curr = eigspec_curr['d1Hnm']; nac1_curr = eigspec_curr['nac1']
        d2En_curr = eigspec_curr['d2En']; d2Hnm_curr = eigspec_curr['d2Hnm']

        C_prev = Cijn_prev.reshape(ndvr**2, nbo); M_prev = np.abs(C_prev); A_prev = np.angle(C_prev)
        C_curr = Cijn_curr.reshape(ndvr**2, nbo); M_curr = np.abs(C_curr); A_curr = np.angle(C_curr)
        Tn_curr = -0.5 * np.sum((M_curr**2 + M_prev**2) * ((A_curr - A_prev + np.pi) % (2*np.pi) - np.pi), axis=0)

        Cijn_curr = Cijn_curr * np.exp(1j * Tn_curr[None,None,:])
        d1Hnm_curr = d1Hnm_curr * np.exp(-1j * (Tn_curr[:,None] - Tn_curr[None,:]))
        nac1_curr = nac1_curr * np.exp(-1j * (Tn_curr[:,None] - Tn_curr[None,:]))
        d2Hnm_curr = d2Hnm_curr * np.exp(-1j * (Tn_curr[:,None] - Tn_curr[None,:]))
        np.savez(sub_dir_curr + "/eigspec2", xi=xi_curr, En=En_curr, Cijn=Cijn_curr, d1En=d1En_curr, d1Hnm=d1Hnm_curr, nac1=nac1_curr, d2En=d2En_curr, d2Hnm=d2Hnm_curr)
        if os.path.isfile(sub_dir_curr + "/boops.npz"):
            boops = {}
            boops_curr = np.load(sub_dir_curr + "/boops.npz")
            for key in boops_curr.files:
                boops[key] = boops_curr[key] * np.exp(-1j * (Tn_curr[:,None] - Tn_curr[None,:]))
            np.savez(sub_dir_curr +"/boops2", **boops)

    return None

def save_grid():
    if not os.path.exists("Rk"):
        print("Missing grid data directory Rk")
        exit()
    kpts = int(os.popen("wc -l < Rk/Rk.dat").read().strip()) - 1
    Rk = np.loadtxt("Rk/Rk.dat", dtype=np.float64)

    sub_dir_0 = os.path.join("Rk", f"R0")
    #eigspec_0 = np.load(os.path.join(sub_dir_0, "eigspec.npz"))
    eigspec_0 = np.load(os.path.join(sub_dir_0, "eigspec_sorted.npz"))
    xi = eigspec_0['xi']; En_0 = eigspec_0['En']; Cijn_0 = eigspec_0['Cijn']
    ndvr = Cijn_0.shape[0]
    nbo = Cijn_0.shape[2]
    Enk = np.zeros((nbo, kpts+1), dtype=np.float64)
    Enk[:,0] = En_0
    if "d1En" in eigspec_0:
        d1En_0 = eigspec_0['d1En']
        d1Enk = np.zeros((nbo, kpts+1), dtype=np.float64)
        d1Enk[:,0] = d1En_0

    for k in range(1, kpts + 1):
        sub_dir_k = os.path.join("Rk", f"R{k}")
        #eigspec_k = np.load(os.path.join(sub_dir_k, "eigspec.npz"))
        eigspec_k = np.load(os.path.join(sub_dir_k, "eigspec_sorted.npz"))
        En_k = eigspec_k['En']; Cijn_k = eigspec_k['Cijn']
        Enk[:,k] = En_k
        if "d1En" in eigspec_k:
            d1En_k = eigspec_k['d1En']
            d1Enk[:,k] = d1En_k

    np.savez('eigspecgrid', xi=xi, Rk=Rk, Enk=Enk)

    return None

def create_eta_grid(coef, delta, lpts, template_file):
    os.makedirs("etal", exist_ok=True)
    etal = coef * (delta**np.arange(lpts+1)-1) / (delta - 1)
    np.savetxt(os.path.join("etal", "etal.dat"), etal)

    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' not found.")
        return

    with open(template_file, "r") as f:
        template_content = f.read()

    for l, eta in enumerate(etal):
        sub_dir = os.path.join("etal", f"eta{l}")
        os.makedirs(sub_dir, exist_ok=True)

        modified_content = template_content.replace("eta_sub", f"{eta:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(template_file)), "w") as f:
            f.write(modified_content)

    return None

def sort_eta_grid():
    if not os.path.exists("etal"):
        print("Missing grid data directory etal")
        exit()
    etal = np.loadtxt("etal/etal.dat", dtype=np.float64)

    for l, eta in enumerate(etal[1:], start=1):
        sub_dir_prev = f"etal/eta{l-1}"
        sub_dir_curr = f"etal/eta{l}"
        capspec_prev = np.load(sub_dir_prev + "/capspec.npz")
        capspec_curr = np.load(sub_dir_curr + "/capspec.npz")
        Cnml_prev = capspec_prev["Cnml"]; Cnmr_prev = capspec_prev["Cnmr"]
        En_curr = capspec_curr["En"] ; Cnml_curr = capspec_curr["Cnml"]; Cnmr_curr = capspec_curr["Cnmr"]

        ovlp = np.matmul(Cnml_prev.conj().T, Cnmr_curr)
        row_ind, col_ind = linear_sum_assignment(-np.absolute(ovlp)**2)
        #col_ind = fix_index_array(np.argmax(np.absolute(ovlp), axis=1))[0]
        print(np.where(row_ind != col_ind))
        np.savez(sub_dir_curr + "/capspec", En=En_curr[col_ind], Cnml=Cnml_curr[:,col_ind], Cnmr=Cnmr_curr[:,col_ind])

    return None

#def sort_R_grid_old():
#    if not os.path.exists("Rk"):
#        print("Missing grid data directory Rk")
#        exit()
#    kpts = int(os.popen("wc -l < Rk/Rk.dat").read().strip()) - 1
#
#    for k in range(1, kpts + 1):
#        sub_dir_prev = os.path.join("Rk", f"R{k-1}")
#        sub_dir_curr = os.path.join("Rk", f"R{k}")
#        #eigspec_prev = np.load(os.path.join(sub_dir_prev, "eigspec.npz"))
#        eigspec_curr = np.load(os.path.join(sub_dir_curr, "eigspec.npz"))
#        eigspec_prev = np.load(os.path.join(sub_dir_prev, "eigspec2.npz"))
#        #eigspec_curr = np.load(os.path.join(sub_dir_curr, "eigspec2.npz"))
#        xi_prev = eigspec_prev['xi']; En_prev = eigspec_prev['En']; Cijn_prev = eigspec_prev['Cijn']
#        xi_curr = eigspec_curr['xi']; En_curr = eigspec_curr['En']; Cijn_curr = eigspec_curr['Cijn']
#
##        ndvr = Cijn_prev.shape[0]
#        nbo = Cijn_prev.shape[2]
#
#        Ck = Cijn_prev.reshape(ndvr**2, nbo); Mk = np.abs(Ck); Ak = np.angle(Ck)
#        Ckp1 = Cijn_curr.reshape(ndvr**2, nbo); Mkp1 = np.abs(Ckp1); Akp1 = np.angle(Ckp1)
#        Tkp1 = -0.5 * np.sum((Mkp1**2 + Mk**2) * (Akp1 - Ak), axis=0)
#        #print(Akp1 / np.pi)
#
#        #print(0.5 * np.sum((Mkp1**2 + Mk**2), axis=0))
#
#        Cijn_curr = Cijn_curr * np.exp(1j * Tkp1[None,None,:])
#
#        if "d2En" in eigspec_curr:
#            #d1En_prev = eigspec_prev['d1En']; d1Hnm_prev = eigspec_prev['d1Hnm']; nac1_prev = eigspec_prev['nac1']
#            d1En_curr = eigspec_curr['d1En']; d1Hnm_curr = eigspec_curr['d1Hnm']; nac1_curr = eigspec_curr['nac1']
#            #d2En_prev = eigspec_prev['d2En']; d2Hnm_prev = eigspec_prev['d2Hnm']
#            d2En_curr = eigspec_curr['d2En']; d2Hnm_curr = eigspec_curr['d2Hnm']
#            d1Hnm_curr = d1Hnm_curr * np.exp(-1j * (Tkp1[:,None] - Tkp1[None,:]))
#            nac1_curr = nac1_curr * np.exp(-1j * (Tkp1[:,None] - Tkp1[None,:]))
#            d2Hnm_curr = d2Hnm_curr * np.exp(-1j * (Tkp1[:,None] - Tkp1[None,:]))
#            #np.savez(os.path.join(sub_dir_curr,"eigspec"), xi=xi_curr, En=En_curr, Cijn=Cijn_curr, d1En=d1En_curr, d1Hnm=d1Hnm_curr, nac1=nac1_curr, d2En=d2En_curr, d2Hnm=d2Hnm_curr)
#            np.savez(os.path.join(sub_dir_curr,"eigspec2"), xi=xi_curr, En=En_curr, Cijn=Cijn_curr, d1En=d1En_curr, d1Hnm=d1Hnm_curr, nac1=nac1_curr, d2En=d2En_curr, d2Hnm=d2Hnm_curr)
#        elif "d1En" in eigspec_curr:
#            #d1En_prev = eigspec_prev['d1En']; d1Hnm_prev = eigspec_prev['d1Hnm']; nac1_prev = eigspec_prev['nac1']
#            d1En_curr = eigspec_curr['d1En']; d1Hnm_curr = eigspec_curr['d1Hnm']; nac1_curr = eigspec_curr['nac1']
#            d1Hnm_curr = d1Hnm_curr * np.exp(-1j * (Tkp1[:,None] - Tkp1[None,:]))
#            nac1_curr = nac1_curr * np.exp(-1j * (Tkp1[:,None] - Tkp1[None,:]))
#            #np.savez(os.path.join(sub_dir_curr,"eigspec"), xi=xi_curr, En=En_curr, Cijn=Cijn_curr, d1En=d1En_curr, d1Hnm=d1Hnm_curr, nac1=nac1_curr)
#            np.savez(os.path.join(sub_dir_curr,"eigspec2"), xi=xi_curr, En=En_curr, Cijn=Cijn_curr, d1En=d1En_curr, d1Hnm=d1Hnm_curr, nac1=nac1_curr)
#        else:
#            #np.savez(os.path.join(sub_dir_curr,"eigspec"), xi=xi_curr, En=En_curr, Cijn=Cijn_curr)
#            np.savez(os.path.join(sub_dir_curr,"eigspec2"), xi=xi_curr, En=En_curr, Cijn=Cijn_curr)
#
#        #d1En=d1En, d1Hnm=d1Hnm, nac1=nac1
#
#        # DONT FORGET THAT THIS WAS THE WORKING CODE
#        #ovlp = np.matmul(Cijn_prev.reshape(ndvr**2, nsort).T, Cijn_curr.reshape(ndvr**2, nbo))
#        #row_ind, col_ind = linear_sum_assignment(-np.absolute(ovlp)**2)
#
#        #print(col_ind)
#        #print(np.where(row_ind != col_ind))
#
#        #np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[col_ind], Cijn=Cijn_curr[:,:,col_ind])
#        #np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[:nsort], Cijn=Cijn_curr[:,:,:nsort])
#        #np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[:nsort], Cijn=Cijn_curr[:,:,:nsort])
#
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_prev.reshape(ndvr**2, nbo))
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_prev[:,:,:nsort].reshape(ndvr**2, nsort))
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_prev.reshape(ndvr**2, nsort))
#        #ovlp = np.matmul(Cijn_prev.reshape(ndvr**2, nsort).T, Cijn_prev.reshape(ndvr**2, nsort))
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_curr.reshape(ndvr**2, nbo))
#        #idx = np.argmax(np.absolute(ovlp), axis=0)
#        #print(np.argwhere(idx==75))
#        #print(ovlp[75,75])
#        #print(ovlp[75,73])
#        #print(ovlp[73,75])
#        #print(ovlp[75,70:80])
#        #print(ovlp[70:80,75])
#        #print(En_prev[73],En_prev[75])
#        #print(sub_dir_prev)
#        #print(En_curr[73],En_curr[75])
#        print(sub_dir_curr)
#        #if (np.unique(idx).size != idx.size):
#        #    print("DUPLICATE")
#        #np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[idx], Cijn=Cijn_curr[:,:,idx])
#
#        #print(np.argmax(np.absolute(ovlp), axis=0))
#        #print(np.argmax(np.absolute(ovlp), axis=1))
#
#        #ovlp = np.matmul(Clnm_prev.conj().T, Crnm_curr)
#        ##idx = np.argmax(np.absolute(ovlp), axis=1)
#        #idx = fix_index_array(np.argmax(np.absolute(ovlp), axis=1))[0]
#        #Er_sort = Er_curr[idx]
#        #Gam_sort = Gam_curr[idx]
#        #Clnm_sort = Clnm_curr[:,idx]
#        #Crnm_sort = Crnm_curr[:,idx]
#        #np.savetxt(os.path.join(sub_dir_curr, "Er.dat"), Er_sort)
#        #np.savetxt(os.path.join(sub_dir_curr, "Gam.dat"), Gam_sort)
#        #np.savetxt(os.path.join(sub_dir_curr, "Clnm.dat"), Clnm_sort)
#        #np.savetxt(os.path.join(sub_dir_curr, "Crnm.dat"), Crnm_sort)
#        #print(idx)
#        #if idx.size != np.unique(idx).size:
#        #    print('DUPLICATE ABOVE!!!')
#        #    vals, counts = np.unique(idx, return_counts=True)
#        #    print(vals[counts > 1])
#
#    return None




#def sort_eta_grid_oldish():
#    if not os.path.exists("etal"):
#        print("Missing grid data directory etal")
#        exit()
#    etal = np.loadtxt("etal/etal.dat", dtype=np.float64)
#
#    for l, eta in enumerate(etal):
#        sub_dir_prev = f"etal/eta{l-1}"
#        sub_dir_curr = f"etal/eta{l}"
#        Clnm_prev = np.loadtxt(os.path.join(sub_dir_prev, "Clnm.dat"), dtype=np.complex128)
#        Crnm_prev = np.loadtxt(os.path.join(sub_dir_prev, "Crnm.dat"), dtype=np.complex128)
#        Er_curr= np.loadtxt(os.path.join(sub_dir_curr, "Er.dat"))
#        Gam_curr = np.loadtxt(os.path.join(sub_dir_curr, "Gam.dat"))
#        Clnm_curr = np.loadtxt(os.path.join(sub_dir_curr, "Clnm.dat"), dtype=np.complex128)
#        Crnm_curr = np.loadtxt(os.path.join(sub_dir_curr, "Crnm.dat"), dtype=np.complex128)
#        #ovlp = np.einsum("ijn,ijm->nm", Cijn_prev, Cijn_curr)
#        #ovlp = np.absolute(np.tensordot(Cijn_prev, Cijn_curr, axes=([0,1],[0,1])))
#        #ovlp = np.matmul(Cnm_prev.conj().T, Cnm_curr)
#        ovlp = np.matmul(Clnm_prev.conj().T, Crnm_curr)
#        #idx = np.argmax(np.absolute(ovlp), axis=1)
#        idx = fix_index_array(np.argmax(np.absolute(ovlp), axis=1))[0]
#        Er_sort = Er_curr[idx]
#        Gam_sort = Gam_curr[idx]
#        Clnm_sort = Clnm_curr[:,idx]
#        Crnm_sort = Crnm_curr[:,idx]
#        np.savetxt(os.path.join(sub_dir_curr, "Er.dat"), Er_sort)
#        np.savetxt(os.path.join(sub_dir_curr, "Gam.dat"), Gam_sort)
#        np.savetxt(os.path.join(sub_dir_curr, "Clnm.dat"), Clnm_sort)
#        np.savetxt(os.path.join(sub_dir_curr, "Crnm.dat"), Crnm_sort)
#        print(idx)
#        exit()
#    kpts = int(os.popen("wc -l < Rk/Rk.dat").read().strip()) - 1
#
#    sub_dir_0 = os.path.join("Rk", f"R0")
#    eigspec_0 = np.load(os.path.join(sub_dir_0, "eigspec.npz"))
#    xi = eigspec_0['xi']; En_0 = eigspec_0['En']; Cijn_0 = eigspec_0['Cijn']
#    np.savez(os.path.join(sub_dir_0,"eigspec_sorted"), xi=xi, En=En_0[:nsort], Cijn=Cijn_0[:,:,:nsort])
#
#    for k in range(1, kpts + 1):
#        sub_dir_prev = os.path.join("Rk", f"R{k-1}")
#        sub_dir_curr = os.path.join("Rk", f"R{k}")
#        #eigspec_prev = np.load(os.path.join(sub_dir_prev, "eigspec.npz"))
#        eigspec_prev = np.load(os.path.join(sub_dir_prev, "eigspec_sorted.npz"))
#        eigspec_curr = np.load(os.path.join(sub_dir_curr, "eigspec.npz"))
#        En_prev = eigspec_prev['En']; Cijn_prev = eigspec_prev['Cijn']
#        En_curr = eigspec_curr['En']; Cijn_curr = eigspec_curr['Cijn']
#        ndvr = Cijn_curr.shape[0]
#        nbo = Cijn_curr.shape[2]
#
#        # DONT FORGET THAT THIS WAS THE WORKING CODE
#        #ovlp = np.matmul(Cijn_prev.reshape(ndvr**2, nsort).T, Cijn_curr.reshape(ndvr**2, nbo))
#        #row_ind, col_ind = linear_sum_assignment(-np.absolute(ovlp)**2)
#
#        #print(col_ind)
#        #print(np.where(row_ind != col_ind))
#
#        #np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[col_ind], Cijn=Cijn_curr[:,:,col_ind])
#        np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[:nsort], Cijn=Cijn_curr[:,:,:nsort])
#        #np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[:nsort], Cijn=Cijn_curr[:,:,:nsort])
#
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_prev.reshape(ndvr**2, nbo))
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_prev[:,:,:nsort].reshape(ndvr**2, nsort))
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_prev.reshape(ndvr**2, nsort))
#        #ovlp = np.matmul(Cijn_prev.reshape(ndvr**2, nsort).T, Cijn_prev.reshape(ndvr**2, nsort))
#        #ovlp = np.matmul(Cijn_curr.reshape(ndvr**2, nbo).T, Cijn_curr.reshape(ndvr**2, nbo))
#        #idx = np.argmax(np.absolute(ovlp), axis=0)
#        #print(np.argwhere(idx==75))
#        #print(ovlp[75,75])
#        #print(ovlp[75,73])
#        #print(ovlp[73,75])
#        #print(ovlp[75,70:80])
#        #print(ovlp[70:80,75])
#        #print(En_prev[73],En_prev[75])
#        #print(sub_dir_prev)
#        #print(En_curr[73],En_curr[75])
#        print(sub_dir_curr)
#        #if (np.unique(idx).size != idx.size):
#        #    print("DUPLICATE")
#        #np.savez(os.path.join(sub_dir_curr,"eigspec_sorted"), xi=xi, En=En_curr[idx], Cijn=Cijn_curr[:,:,idx])
#
#        #print(np.argmax(np.absolute(ovlp), axis=0))
#        #print(np.argmax(np.absolute(ovlp), axis=1))
#
#        #ovlp = np.matmul(Clnm_prev.conj().T, Crnm_curr)
#        ##idx = np.argmax(np.absolute(ovlp), axis=1)
#        #idx = fix_index_array(np.argmax(np.absolute(ovlp), axis=1))[0]
#        #Er_sort = Er_curr[idx]
#        #Gam_sort = Gam_curr[idx]
#        #Clnm_sort = Clnm_curr[:,idx]
#        #Crnm_sort = Crnm_curr[:,idx]
#        #np.savetxt(os.path.join(sub_dir_curr, "Er.dat"), Er_sort)
#        #np.savetxt(os.path.join(sub_dir_curr, "Gam.dat"), Gam_sort)
#        #np.savetxt(os.path.join(sub_dir_curr, "Clnm.dat"), Clnm_sort)
#        #np.savetxt(os.path.join(sub_dir_curr, "Crnm.dat"), Crnm_sort)
#        #print(idx)
#        #if idx.size != np.unique(idx).size:
#        #    print('DUPLICATE ABOVE!!!')
#        #    vals, counts = np.unique(idx, return_counts=True)
#        #    print(vals[counts > 1])
#
#    return None

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
