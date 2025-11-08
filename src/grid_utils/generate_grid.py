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

