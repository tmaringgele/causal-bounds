from .base_iv import IVScenario
from .binary_iv import BinaryIV
import numpy as np
import pandas as pd
from simulation_engine.util.datagen_util import datagen_util
from scipy.stats import norm


class ContinuousIV(IVScenario):
    def __init__(self, dag, dataframe_cont, bin_size=10):
        super().__init__(dag)
        self.bin_size = bin_size
        binned_data = self._bin_data(dataframe_cont)
        # Create internal binary IV object
        self.binary_iv_scenario = BinaryIV(dag, binned_data)

    def _bin_data(self, dataframe_cont):
        print("Binning continuous data with bin size:", self.bin_size, 'from:', dataframe_cont)
        return 'binned data'  
    

    


    
    @staticmethod
    def run_continuous_iv_simulations(N=2000, n=500, seed=None, allowed_functions=None):
        results = []
        for i in range(N):
            sim_seed = seed + i if seed is not None else None
            result = ContinuousIV.generate_data(n=n, seed=sim_seed, allowed_functions=allowed_functions)
            results.append(result)
        return pd.DataFrame(results)
    @staticmethod
    def run_rolling_b_X_Y_simulations(
        b_range=(-5, 5),
        N_points=50,
        replications=20,
        n=500,
        seed=None,
        allowed_functions=None
    ):
        """
        Run simulations across a range of b_X_Y values with multiple replications per point,
        returning a DataFrame with all individual simulation results (not aggregated).

        Args:
            b_range (tuple): Range (min, max) of b_X_Y values.
            N_points (int): Number of evenly spaced b_X_Y values to try.
            replications (int): Number of replications per b_X_Y value.
            n (int): Sample size per simulation.
            seed (int or None): Optional base seed.
            allowed_functions (list of str or None): If specified, restricts function choices.

        Returns:
            pd.DataFrame: All simulation results with repeated b_X_Y values.
        """
        all_results = []
        b_values = np.linspace(b_range[0], b_range[1], N_points)

        for i, b_X_Y_val in enumerate(b_values):
            for j in range(replications):
                sim_seed = (seed + i * replications + j) if seed is not None else None
                result = ContinuousIV.generate_data(
                    n=n,
                    seed=sim_seed,
                    b_X_Y=b_X_Y_val,
                    allowed_functions=allowed_functions
                )
                result['b_X_Y'] = b_X_Y_val
                all_results.append(result)

        return pd.DataFrame(all_results)


    
    @staticmethod
    def generate_data(
        n=500,
        seed=None,
        sigma_X=None,
        sigma_Y=None,
        b_Z_X=None,
        b_U_X=None,
        b_X_Y=None,
        b_U_Y=None,
        allowed_functions=None
    ):
        if seed is None:
            seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        # Function dictionary
        G_all = {
            "identity": lambda x: x,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "log1p_abs": lambda x: np.log1p(np.abs(x)),
            "exp_neg_sq": lambda x: np.exp(-x**2),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "exp_clipped": lambda x: np.exp(np.clip(x, -5, 5))
        }

        G = {k: v for k, v in G_all.items() if allowed_functions is None or k in allowed_functions}

        # Coefficients
        if sigma_X is None:
            sigma_X = np.abs(np.random.normal(0, 1))
        if sigma_Y is None:
            sigma_Y = np.abs(np.random.normal(0, 1))
        if b_Z_X is None:
            b_Z_X = datagen_util.pick_from_bimodal()
        if b_U_X is None:
            b_U_X = np.random.normal(0, 1)
        if b_X_Y is None:
            b_X_Y = np.random.normal(0, 1)
        if b_U_Y is None:
            b_U_Y = np.random.normal(0, 1)

        # Random nonlinearities
        g_Z_X_name = np.random.choice(list(G.keys()))
        g_U_X_name = np.random.choice(list(G.keys()))
        g_X_Y_name = np.random.choice(list(G.keys()))
        g_U_Y_name = np.random.choice(list(G.keys()))

        g_Z_X = G[g_Z_X_name]
        g_U_X = G[g_U_X_name]
        g_X_Y = G[g_X_Y_name]
        g_U_Y = G[g_U_Y_name]

        # Binary instrument and confounder
        Z = np.random.binomial(1, 0.5, n)
        U = np.random.binomial(1, 0.5, n)

        # Latent X and stochastic binarization
        X_latent = b_Z_X * g_Z_X(Z) + b_U_X * g_U_X(U) + np.random.normal(0, sigma_X, n)

        # Squashing functions to map latent X to [0,1] for Bernoulli sampling
        squashers = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh_scaled": lambda x: 0.5 * (1 + np.tanh(x)),
            "softplus": lambda x: (np.log1p(np.exp(x))) / (1 + np.log1p(np.exp(x))),
            "probit": lambda x: norm.cdf(x),
        }

        squash_name = np.random.choice(list(squashers.keys()))
        squasher = squashers[squash_name]
        p_X = squasher(X_latent)
        X = np.random.binomial(1, p_X)

        # Continuous outcome (clipped to [-100, 100])
        epsilon_Y = np.random.normal(0, sigma_Y, n)
        Y_raw = b_X_Y * g_X_Y(X) + b_U_Y * g_U_Y(U) + epsilon_Y
        Y = np.clip(Y_raw, -20, 20)

        # Counterfactual outcomes
        Y1 = np.clip(b_X_Y * g_X_Y(1) + b_U_Y * g_U_Y(U) + np.random.normal(0, sigma_Y, n), -20, 20)
        Y0 = np.clip(b_X_Y * g_X_Y(0) + b_U_Y * g_U_Y(U) + np.random.normal(0, sigma_Y, n), -20, 20)

        ATE_true = np.mean(Y1 - Y0)
        PNS_true = np.mean((Y1 > Y0).astype(float))

        return {
            'seed': seed,
            'b_Z_X': b_Z_X,
            'b_U_X': b_U_X,
            'b_X_Y': b_X_Y,
            'b_U_Y': b_U_Y,
            'sigma_X': sigma_X,
            'sigma_Y': sigma_Y,
            'g_Z_X': g_Z_X_name,
            'g_U_X': g_U_X_name,
            'g_X_Y': g_X_Y_name,
            'g_U_Y': g_U_Y_name,
            'squash_X': squash_name,
            'ATE_true': ATE_true,
            'PNS_true': PNS_true,
            'p_Y1_mean': np.mean(Y1),
            'p_Y0_mean': np.mean(Y0),
            'Z': Z,
            'U': U,
            'X': X,
            'Y': Y,
            'Y_max': np.max(Y),
            'Y_min': np.min(Y),
            'X_max': np.max(X),
            'X_min': np.min(X),
            'X_mean': np.mean(X),
            'Y_mean': np.mean(Y),
        }
