from grid_utils.generate_grid import create_eta_grid

coef = 5.0e-7
delta = 1.1
npts = 49
#npts = 125

create_eta_grid(coef, delta, npts, "template.py")

