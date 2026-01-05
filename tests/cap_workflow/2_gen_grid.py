from grid_utils.generate_grid import create_eta_grid

#coef = 7.5e-6
#delta = 1.1
#npts = 45

coef = 1.0e-7
delta = 1.2
npts = 75

create_eta_grid(coef, delta, npts, "template.py")

