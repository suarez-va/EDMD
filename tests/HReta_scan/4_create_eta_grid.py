import os
from grids.grid_utils import create_eta_grid

coef = 1.0e-8
delta = 1.125
lpts = 150
#coef = 1.0e-7
#delta = 1.1 #delta = 1.2
#lpts = 150 #lpts = 75

if not os.path.exists("Rk"):
    print("Missing grid data directory Rk")
    exit()
kpts = int(os.popen("wc -l < Rk/Rk.dat").read().strip()) - 1

for k in range(kpts + 1):
    os.chdir(f"Rk/R{k}")
    create_eta_grid(coef, delta, lpts, "../../template_eta.py")
    os.chdir("../../")


