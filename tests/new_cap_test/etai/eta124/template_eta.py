import numpy as np
from time_independent.grid_utils import solve_cap

eta = 0.17619864865

fcidvr_data = np.load('../../fcidvr_2ele_singlet.npz')
cap_data = np.load('../../fcidvr_2ele_singlet_cap.npz')

Hnm = np.diag(fcidvr_data['En'])
Wnm = cap_data['Wnm']
En, Cnml, Cnmr = solve_cap(Hnm, Wnm, 0.17619864865, 'fcicap_2ele_singlet')


