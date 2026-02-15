import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, model_params: dict):
        self.params = model_params

    @abstractmethod
    def VR(self, R):
        pass

    @abstractmethod
    def d1VR(self, R):
        pass

    @abstractmethod
    def d2VR(self, R):
        pass

    @abstractmethod
    def VeR(self, x, R):
        pass

    @abstractmethod
    def d1VeR(self, x, R):
        pass

    @abstractmethod
    def d2VeR(self, x, R):
        pass

    @abstractmethod
    def Vee(self, x1, x2):
        pass


class GICD(Model):
    def __init__(self, model_params: dict):
        super().__init__(model_params)

    def VR(self, R):
        ZA = self.params["ZA"]
        ZB = self.params["ZB"]
        aR = self.params["aR"]
        bR = self.params["bR"]
        return ZA * ZB * np.exp(-aR * R**2) / np.sqrt(R**2 + bR)

    def d1VR(self, R):
        aR = self.params["aR"]
        bR = self.params["bR"]
        GR = -R * (2 * aR + 1 / (R**2 + bR))
        return GR * self.VR(R)

    def d2VR(self, R):
        aR = self.params["aR"]
        bR = self.params["bR"]
        GR = -R * (2 * aR + 1 / (R**2 + bR))
        dGR = -(2 * aR + 1 / (R**2 + bR)) + 2 * R**2 / ((R**2 + bR)**2)
        return (dGR + GR**2) * self.VR(R)

    def VeR(self, x, R):
        DAe = self.params["DA"]
        bAe = self.params["bA"]
        DBe = self.params["DB"]
        bBe = self.params["bB"]
        Aarg = (x + R / 2); Barg = (x - R / 2)
        return -DAe * np.exp(-bAe * Aarg**2) - DBe * np.exp(-bBe * Barg**2)

    def d1VeR(self, x, R):
        DAe = self.params["DA"]
        bAe = self.params["bA"]
        DBe = self.params["DB"]
        bBe = self.params["bB"]
        Aarg = (x + R / 2); Barg = (x - R / 2)
        return DAe * bAe * Aarg * np.exp(-bAe * Aarg**2) - DBe * bBe * Barg * np.exp(-bBe * Barg**2)

    def d2VeR(self, x, R):
        DAe = self.params["DA"]
        bAe = self.params["bA"]
        DBe = self.params["DB"]
        bBe = self.params["bB"]
        Aarg = (x + R / 2); Barg = (x - R / 2)
        return DAe * bAe * (0.5 - bAe * Aarg**2) * np.exp(-bAe * Aarg**2) + DBe * bBe * (0.5 - bBe * Barg**2) * np.exp(-bBe * Barg**2)
        #return 0.5 * (DAe * bAe * (1 - bAe * Aarg**2) * np.exp(-bAe * Aarg**2) + DBe * bBe * (1 - bBe * Barg**2) * np.exp(-bBe * Barg**2))

    def Vee(self, x1, x2):
        aee = self.params["aee"]
        bee = self.params["bee"]
        xarg = (x1 - x2)**2
        return np.exp(-aee * xarg) / np.sqrt(xarg + bee)


#class TESD(Model):
#    def __init__(self, a: float, b: float, N: int, bounds: str, spin: str, model_params: dict):
#        super().__init__(a, b, N, bounds, spin, model_params)
#
#    def VR(self, R):
#        ZA = self.params["ZA"]
#        ZB = self.params["ZB"]
#        aR = self.params["aR"]
#        bR = self.params["bR"]
#        return ZA * ZB * np.exp(-aR * R**2) / np.sqrt(R**2 + bR)
#
#    def dVR(self, R):
#        ZA = self.params["ZA"]
#        ZB = self.params["ZB"]
#        aR = self.params["aR"]
#        bR = self.params["bR"]
#        return -ZA * ZB * (2 * aR + 1 / (R**2 + bR)) * R * np.exp(-aR * R**2) / np.sqrt(R**2 + bR)
#
#    def VeR(self, x, R):
#        ZA = self.params["ZA"]
#        ZB = self.params["ZB"]
#        aAe = self.params["aAe"]
#        bAe = self.params["bAe"]
#        aBe = self.params["aBe"]
#        bBe = self.params["bBe"]
#        mA = self.params["mA"]
#        mB = self.params["mB"]
#        mu = mA * mB / (mA + mB)
#        Aarg = (x + mu / mA * R); Barg = (x - mu / mB * R)
#        return -ZA * np.exp(-aAe * Aarg**2) / np.sqrt(Aarg**2 + bAe) - ZB * np.exp(-aBe * Barg**2) / np.sqrt(Barg**2 + bBe)
#
#    def dVeR(self, x, R):
#        ZA = self.params["ZA"]
#        ZB = self.params["ZB"]
#        aAe = self.params["aAe"]
#        bAe = self.params["bAe"]
#        aBe = self.params["aBe"]
#        bBe = self.params["bBe"]
#        mA = self.params["mA"]
#        mB = self.params["mB"]
#        mu = mA * mB / (mA + mB)
#        Aarg = (x + mu / mA * R); Barg = (x - mu / mB * R)
#        return ZA * mu / mA * (2 * aAe + 1 / (Aarg**2 + bAe)) * Aarg * np.exp(-aAe * Aarg**2) / np.sqrt(Aarg**2 + bAe) - ZB * mu / mB * (2 * aBe + 1 / (Barg**2 + bBe)) * Barg * np.exp(-aBe * Barg**2) / np.sqrt(Barg**2 + bBe)
#
#    def Vee(self, x1, x2):
#        aee = self.params["aee"]
#        bee = self.params["bee"]
#        xarg = (x1 - x2)**2
#        return np.exp(-aee * xarg) / np.sqrt(xarg + bee)
#        #return 0 * np.exp(-aee * xarg) / np.sqrt(xarg + bee)


