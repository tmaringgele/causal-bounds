from simulation_engine.algorithms.autobound import AutoBound
from simulation_engine.scenarios.scenario import Scenario
import pandas as pd
import numpy as np


class CustomBinaryDAG(Scenario):
    """
    A Scenario that can run various algorithms on a custom DAG.
    """

    AVAILABLE_ALGORITHMS = {
        "ATE_autobound": lambda self: AutoBound.bound(
            query='ATE',
            data=self.data,
            dagstring=self.dagstring,
            observed=self.observed,
            unob=self.unobserved,
            indep=self.indep,
            dep=self.dep
        ),
        "PNS_autobound": lambda self: AutoBound.bound(
            query='PNS',
            data=self.data,
            dagstring=self.dagstring,
            observed=self.observed,
            unob=self.unobserved,
            indep=self.indep,
            dep=self.dep
        )
    }

    def __init__(self, dataframe, dagstring="X -> Y, U -> X, U -> Y", unobserved=['U'], indep='X', dep='Y'):
        super().__init__()
        self.data = dataframe
        self.dagstring = dagstring
        self.unobserved = unobserved
        self.observed = CustomBinaryDAG._getObserved(dagstring, unobserved)
        self.indep = indep
        self.dep = dep

    @staticmethod
    def _getObserved(dagstring, unobserved):
        """
        Extracts the observed variables from the DAG string.
        Parameters:
            dagstring (str): The string representation of the DAG.
            unobserved (list): List of unobserved variable names.
        Returns:
            list: List of observed variables.
        """
        variables = set()
        for edge in dagstring.split(','):
            src, dst = [v.strip() for v in edge.strip().split('->')]
            if src not in unobserved:
                variables.add(src)
            if dst not in unobserved:
                variables.add(dst)
        return list(variables)

    @staticmethod
    def generateMediation(n=10):
        """
        Generate a mediation scenario.
        Returns:
            df: DataFrame with columns X, M, Y
            ATE_true: The true average treatment effect from X to Y
        """
        X = np.random.binomial(1, 0.5, n)
        M = np.random.binomial(1, 0.5, n)
        Y_star = 0.7 * X + 0.3 * M

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        Y = np.random.binomial(1, sigmoid(Y_star), n)
        df = pd.DataFrame({'X': X, 'M': M, 'Y': Y})

        # Compute true ATE: E[Y|do(X=1)] - E[Y|do(X=0)]
        # Under the data generating process, M is independent of X, so P(M=1)=0.5
        # E[Y|do(X=x)] = E_M[ sigmoid(0.7*x + 0.3*M) ]
        E_Y_do_X1 = 0.5 * sigmoid(0.7*1 + 0.3*0) + 0.5 * sigmoid(0.7*1 + 0.3*1)
        E_Y_do_X0 = 0.5 * sigmoid(0.7*0 + 0.3*0) + 0.5 * sigmoid(0.7*0 + 0.3*1)
        ATE_true = E_Y_do_X1 - E_Y_do_X0

        return df, ATE_true


