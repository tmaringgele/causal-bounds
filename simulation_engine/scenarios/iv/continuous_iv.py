from .base_iv import IVScenario
from .binary_iv import BinaryIV

class ContinuousIV(IVScenario):
    def __init__(self, dag, dataframe_cont, bin_size=10):
        super().__init__(dag)
        self.bin_size = bin_size
        self.binned_data = self._bin_data(dataframe_cont)

        # Create internal binary IV object
        self.binary_iv_scenario = BinaryIV(dag, self.binned_data)

    def _bin_data(self, dataframe_cont):
        print("Binning continuous data with bin size:", self.bin_size, 'from:', dataframe_cont)
        return 'binned data'  


    def bound_ate_causaloptim(self):
        print("Delegating to binary ATE bound using binned data")
        return self.binary_iv_scenario.bound_ate_causaloptim()


    def bound_ate_csm(self):
        print("Boudning ATE using Curvature Sensitivity Model")
        return (0.1, 0.9)

    def plot_continuous_distribution(self):
        import matplotlib.pyplot as plt
        plt.hist(self.continuous_data, bins=self.bin_size)
        plt.title("Distribution of continuous treatment")
        plt.show()