from .base_iv import IVScenario
from .binary_iv import BinaryIV

class ContinuousIV(IVScenario):
    def __init__(self, dag, continuous_data, bin_size=10):
        super().__init__(dag)
        self.continuous_data = continuous_data
        self.bin_size = bin_size
        self.binned_data = self._bin_data(continuous_data)

        # Create internal binary IV object only if needed
        self.binary_iv_scenario = BinaryIV(dag, sample_size=len(self.binned_data))
        self.binary_iv_scenario.data = {"Z": self.binned_data, "D": self.binned_data, "Y": self.binned_data}

    def _bin_data(self, data):
        return [int(x > 0.5) for x in data]

    def generate_data(self):
        print("Generating continuous IV data (e.g., from structural equation)")

    def bound_ate_causaloptim(self, data_override=None):
        print("Delegating to binary ATE bound using binned data")
        return self.binary_iv_scenario.bound_ate_causaloptim(data_override)

    def bound_pns_lp(self, data_override=None):
        print("Delegating to binary PNS bounds using binned data")
        return self.binary_iv_scenario.bound_pns_lp(data_override)

    def estimate_ate_continuous_kernel(self):
        print("Running continuous-only ATE estimator")

    def plot_continuous_distribution(self):
        import matplotlib.pyplot as plt
        plt.hist(self.continuous_data, bins=self.bin_size)
        plt.title("Distribution of continuous treatment")
        plt.show()