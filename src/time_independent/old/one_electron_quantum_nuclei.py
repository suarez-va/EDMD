import numpy as np
from scipy.linalg import eigh
from grids.colbert_miller_dvr import dvr_T
from model_systems.models import Model
from time_independent.fci.one_electron_fixed_nuclei import OneElectronFixedNuclei


class OneElectronQuantumNuclei(OneElectronFixedNuclei):
    def __init__(self, model: Model, xa: float, xb: float, xN: int, xbounds: str, Ra: float, Rb: float, RN: int, Rbounds: str):

        super().__init__(model, xa, xb, xN, xbounds)

        self.Ra = Ra
        self.Rb = Rb
        self.RN = RN
        self.Rbounds = Rbounds
        assert self.Rbounds in ('(a,b)', '(0,inf)', '(-inf,inf)'), f"Colber Miller Nuclear DVR Bounds: {self.Rbounds}, must be '(a,b)', '(0,inf)', or '(-inf,inf)'."

        self.nRdvr = int(self.RN-1) if self.Rbounds=='(a,b)' else int(self.RN) if self.Rbounds=='(0,inf)' else int(self.RN+1) if self.Rbounds=='(-inf,inf)' else 0

    def RI(self) -> np.ndarray:
        return np.linspace(self.Ra, self.Rb, self.nRdvr)

