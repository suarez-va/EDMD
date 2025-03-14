import numpy as np

def dvr_p(dx, a, b, N):
    """

    Generates the Colbert-Miller DVR momentum matrix

    Args:     
        dx ( float ): separation between grid points
        ham_old ( Hamiltonian ): Hamiltonian at time t-dt [units: a.u. of energy]
        ham_cur ( Hamiltonian ): Hamiltonian at time t [units: a.u. of energy]
        orb ( list of ints ): indices of the orbitals included in the active space. The Hvib dimensions will 
            be determined by the N_act = len(orb) and the elements of Hvib will reflect only the orbitals included 
            in this active state. Indexing starts with 0.
        dt ( float ): Time step [units: a.u. of time]

    Returns:
        CMATRIX(N_act,N_act): vibronic Hamiltonian matrix in the MO basis:  Hvib = Hel - i*hbar*d_ij

    """

    return None


