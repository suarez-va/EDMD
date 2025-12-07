from grid_utils.generate_grid import create_eta_grid

coef = 1e-6
delta = 1.12
#npts = 49
npts = 125

create_eta_grid(coef, delta, npts, "template.py")

