from .base_iv import IVScenario
import numpy as np

class BinaryIV(IVScenario):
    def __init__(self, dag, dataframe):
        super().__init__(dag)
        self.data = dataframe


    def bound_ate_causaloptim(self):

        print('running bound_ate_causaloptim')
        print("Data used for ATE calculation:", self.data)
        return (0.2, 0.8)  # placeholder

