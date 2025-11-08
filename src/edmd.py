import os
import numpy as np


# Exact Diatomic Molecular Dynamics
class EDMD:
    def __init__(self, ndim = 3):

        if not os.path.exists("Rk"):
            print("Missing grid data directory Rk")
            exit()
        elif not os.path.isfile(os.path.join("Rk", "Rgrid.dat")):
            print("Missing grid data Rk/Rgrid.dat")
            exit()
        elif not os.path.isfile(os.path.join("Rk", "Eadi.dat")):
            print("Missing grid data Rk/Eadi.dat")
            exit()
        elif not os.path.isfile(os.path.join("Rk", "nac1.dat")):
            print("Missing grid data Rk/nac1.dat")
            exit()
        elif not os.path.isfile(os.path.join("Rk", "nac2.dat")):
            print("Missing grid data Rk/nac2.dat")
            exit()

        Eadi = np.loadtxt(os.path.join("Rk", "Eadi.dat"))
        self.ndim = ndim
        self.Rpts, self.nsta = nsta = Eadi.shape
        self.adiabatic_grid_data = {
        'Rgrid' : np.loadtxt(os.path.join("Rk", "Rgrid.dat")),
        'Eadi' : Eadi,
        'nac1' : np.loadtxt(os.path.join("Rk", "nac1.dat"), dtype=complex).reshape(self.Rpts, self.nsta, self.nsta),
        'nac2' : np.loadtxt(os.path.join("Rk", "nac2.dat"), dtype=complex).reshape(self.Rpts, self.nsta, self.nsta)
        }

        print(self.adiabatic_grid_data['Eadi'])
        print(self.adiabatic_grid_data['nac1'].shape)

#        self.adiabatic_grid_data = {
#        'Eadi' : True,
#        'NAC1' : True,
#        'NAC2' : True
#        }
#
#        rt_scf._operator_functions = {
#        'energy'               : [get_energy, rt_output._print_energy],
#        'fock_ao'              : [lambda *args: None, rt_output._print_fock_ao],
#        }


#    def generate_Hmol(self):
        #self.current_time += self.timestep

        #self.nac1 = np.zeros((Rpts, nstates, nstates), dtype=complex)
        #self.nac2 = np.zeros((Rpts, nstates, nstates), dtype=complex)
        #self.Hmol = np.zeros((self.Rpts * self.nstates, self.Rpts * self.nstates), dtype=complex)


