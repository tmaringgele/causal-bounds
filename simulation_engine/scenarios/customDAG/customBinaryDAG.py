from simulation_engine.algorithms.autobound import AutoBound
from simulation_engine.scenarios.scenario import Scenario
import pandas as pd
import numpy as np


class CustomBinaryDAG(Scenario):
    """
    A Scenario that can run various algorithms on a custom DAG.
    """

    AVAILABLE_ALGORITHMS = {
        "ATE_autobound": lambda self: AutoBound.bound_binaryIV("ATE", self.data, 
                        dagstring=self.dagstring,
                        unob=self.unobserved,
                        )
    }

    def __init__(self, dataframe, dagstring="X -> Y, U -> X, U -> Y", unobserved='U', indep='X', dep='Y'):
        super().__init__()
        self.data = dataframe
        self.dagstring = dagstring
        self.unobserved = unobserved


    @staticmethod
    def generateMediation(n=10):
        """
        Generate a mediation scenario.
        """
        X = np.random.binomial(1, 0.5, n)
        M = np.random.binomial(1, 0.5, n)
        Y_star = 0.7 * X + 0.3 * M

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        Y = np.random.binomial(1, sigmoid(Y_star), n)
        df = pd.DataFrame({'X': X, 'M': M, 'Y': Y})
        return df


