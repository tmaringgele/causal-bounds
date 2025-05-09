from .base_iv import IVScenario
from simulation_engine.util.datagen_util import datagen_util
import numpy as np
import pandas as pd

class BinaryIV(IVScenario):
    def __init__(self, dag, dataframe):
        super().__init__(dag)
        self.data = dataframe

    # enriches self.data with causaloptim bound information
    def bound_ate_causaloptim(self):

        print('running bound_ate_causaloptim')
        print("Data used for ATE calculation:", self.data)
        return (0.2, 0.8)  # placeholder

    

    @staticmethod
    def generate_data(N_simulations=2000, n=500, seed=None, b_U_X=None, b_U_Y=None, b_Z=None, b_X_Y=None, intercept_X=None, intercept_Y=None):
        """
        Generate data for a binary instrumental variable scenario.

        Args:
            N_simulations (int): Number of simulations to run. Default is 2000.
            n (int): Number of samples to generate. Default is 500.
            seed (int, optional): Random seed for reproducibility. Default is None.
            b_U_X (float): Coefficient for the effect of unobserved confounder U on X. Default is drawn from N(0, 1).
            b_U_Y (float): Coefficient for the effect of unobserved confounder U on Y. Default is drawn from N(0, 1).
            b_Z (float): Coefficient for the effect of instrument Z on X. Default is drawn from a bimodal distribution.
            b_X_Y (float): Coefficient for the effect of treatment X on Y. Default is drawn from a bimodal distribution.
            intercept_X (float): Intercept for the logistic model of X. Default is 0.
            intercept_Y (float): Intercept for the logistic model of Y. Default is 0.

        Returns:
            dict: A dictionary containing the generated data and parameters.
        """

        df_results = []
        for i in range(N_simulations):
            result = BinaryIV._simulate_deterministic_data_with_probabilistic_ate(
                n=n,
                seed=seed,
                b_U_X=b_U_X,
                b_U_Y=b_U_Y,
                b_Z=b_Z,
                b_X_Y=b_X_Y,
                intercept_X=intercept_X,
                intercept_Y=intercept_Y
            )
            df_results.append(result)
        return pd.DataFrame(df_results)

    def _simulate_deterministic_data_with_probabilistic_ate(
        n=500,
        seed= None,
        b_U_X=None,
        b_U_Y=None,
        b_Z=None,
        b_X_Y=None,
        intercept_X=None,
        intercept_Y=None,
        p_U = None,
        p_Z = None
    ):
        """
        Simulate deterministic (binary) data for causal analysis, 
        while computing the Average Treatment Effect (ATE) from smooth logistic potential outcome probabilities.

        Args:
            n (int): Number of samples to generate. Default is 500.
            seed (int, optional): Random seed for reproducibility. Default is None.
            b_U_X (float): Coefficient for the effect of unobserved confounder U on X. Default is drawn from N(0, 1).
            b_U_Y (float): Coefficient for the effect of unobserved confounder U on Y. Default is drawn from N(0, 1).
            b_Z (float): Coefficient for the effect of instrument Z on X. Default is drawn from a bimodal distribution.
            b_X_Y (float): Coefficient for the effect of treatment X on Y. Default is drawn from a bimodal distribution.
            intercept_X (float): Intercept for the logistic model of X. Default is 0.
            intercept_Y (float): Intercept for the logistic model of Y. Default is 0.
            p_U (float): Probability of unobserved confounder U ~ Bernoulli(p_U). Default is drawn from Uniform(0, 1).
            p_Z (float): Probability of instrument Z ~ Bernoulli(p_Z). Default is drawn from Uniform(0, 1).

        Returns:
            dict: A dictionary containing:
                - seed (int): The random seed used.
                - intercept_X (float): Intercept for X.
                - intercept_Y (float): Intercept for Y.
                - b_Z (float): Coefficient for Z.
                - b_U_X (float): Coefficient for U on X.
                - b_X_Y (float): Coefficient for X on Y.
                - b_U_Y (float): Coefficient for U on Y.
                - ATE_true (float): True Average Treatment Effect.
                - p_Y1 (np.ndarray): Probabilities of Y=1 under treatment.
                - p_Y0 (np.ndarray): Probabilities of Y=1 under control.
                - p_U (float): Probability of unobserved confounder U.
                - p_Z (float): Probability of instrument Z.
                - Z (np.ndarray): Instrument variable.
                - U (np.ndarray): Unobserved confounder.
                - X (np.ndarray): Treatment assignment.
                - Y (np.ndarray): Outcome variable.
        """
        if seed is None:
            seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        # Assign default values if parameters are None
        if b_U_X is None:
            b_U_X = np.random.normal(0, 1, 1)[0]
        if b_U_Y is None:
            b_U_Y = np.random.normal(0, 1, 1)[0]
        if b_Z is None:
            b_Z = datagen_util.pick_from_bimodal()
        if b_X_Y is None:
            b_X_Y = datagen_util.pick_from_bimodal()
        if intercept_X is None:
            intercept_X = 0
        if intercept_Y is None:
            intercept_Y = 0
        if p_U is None:
            p_U = np.random.uniform(0, 1)
        if p_Z is None:
            p_Z = np.random.uniform(0, 1)

        # Binary variables
        Z = np.random.binomial(1, p_Z, size=n)
        U = np.random.binomial(1, p_U, size=n)

        # Treatment assignment
        logit_X = intercept_X + b_Z * Z + b_U_X * U
        p_X = 1 / (1 + np.exp(-logit_X))
        X = np.random.binomial(1, p_X)

        # Deterministic outcome
        logit_Y = intercept_Y + b_X_Y * X + b_U_Y * U
        p_Y = 1 / (1 + np.exp(-logit_Y))
        Y = np.random.binomial(1, p_Y)

        # Probabilistic potential outcomes
        logit_Y1 = intercept_Y + b_X_Y * 1 + b_U_Y * U
        logit_Y0 = intercept_Y + b_X_Y * 0 + b_U_Y * U
        p_Y1 = 1 / (1 + np.exp(-logit_Y1))
        p_Y0 = 1 / (1 + np.exp(-logit_Y0))
        ATE_true = np.mean(p_Y1 - p_Y0)

        return {
            'seed': seed,
            'intercept_X': intercept_X,
            'intercept_Y': intercept_Y,
            'b_Z': b_Z,
            'b_U_X': b_U_X,
            'b_X_Y': b_X_Y,
            'b_U_Y': b_U_Y,
            'ATE_true': ATE_true,
            'p_Y1_mean': p_Y1.mean(),
            'p_Y0_mean': p_Y0.mean(),
            'p_U': p_U,
            'p_Z': p_Z,
            'Z': Z,
            'U': U,
            'X': X,
            'Y': Y
        }
