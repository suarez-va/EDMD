from grid_utils.generate_grid import create_eta_grid

coef = 1e-4
delta = 1.12
npts = 49
#delta = 1.3
#npts = 5

create_eta_grid(coef, delta, npts, "template.py")

