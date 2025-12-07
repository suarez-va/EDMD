from grid_utils.generate_grid import sort_eta_data, get_eta_data
import numpy as np
import matplotlib.pyplot as plt

#eta_ar, Er_ar, Gam_ar = get_eta_data(16)
eta_ar, Er_ar, Gam_ar = get_eta_data(38)

plt.figure()
npts = eta_ar.shape[0]
for i in range(npts):
    plt.text(Er_ar[i], Gam_ar[i], str(i), ha='center', va='center')
plt.plot(Er_ar[:npts],Gam_ar[:npts],'ro')
#plt.xlim([0.0,10.0])
#plt.ylim([-0.1,0.0])
#plt.ylim([-0.03,0.0])
plt.savefig("traj.png")

