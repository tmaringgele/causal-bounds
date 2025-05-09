from .base_iv import IVScenario
import numpy as np

class BinaryIV(IVScenario):
    def __init__(self, dag, sample_size=1000):
        super().__init__(dag)
        self.sample_size = sample_size
        self.data = None

    def generate_data(self):
        n = self.sample_size
        Z = np.random.binomial(1, 0.5, size=n)
        U = np.random.binomial(1, 0.5, size=n)
        D = (Z + U > 1).astype(int)
        Y = (D + U > 1).astype(int)

        self.data = {"Z": Z, "D": D, "Y": Y, "U": U}

    def bound_ate_causaloptim(self, data_override=None):
        data = data_override if data_override is not None else self.data
        z, y = data["Z"], data["Y"]
        # Simple ATE bound via difference in means
        return np.mean(y[z == 1]) - np.mean(y[z == 0])

    def bound_pns_lp(self, data_override=None):
        data = data_override if data_override is not None else self.data
        # Placeholder for LP-based bounds
        return (0.1, 0.5)