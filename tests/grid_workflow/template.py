from models.two_electron_diatomic import TEGD

params = {
    "aR": 0.0,
    "bR": 0.0001,
    "DA" : 1.0,
    "bA": 0.25,
    "DB" : 0.8,
    "bB": 1.0,
    "aee": 0.0,
    "bee": 0.0001,
}

model = TEGD(a=-25.0, b=25.0, N=80, bounds="(-inf,inf)", spin="triplet", model_params=params)

model.solve_wfn(R = REPLACE, nbo = 125, grad = 2)

