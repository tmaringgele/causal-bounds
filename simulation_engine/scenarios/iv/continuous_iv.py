from .base_iv import IVScenario
from .binary_iv import BinaryIV
import numpy as np
import pandas as pd
from simulation_engine.util.datagen_util import datagen_util

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

        G_all = {
            "identity": lambda x: x,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "log1p_abs": lambda x: np.log(1 + np.abs(x)),
            "exp_neg_sq": lambda x: np.exp(-x**2),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "exp_clipped": lambda x: np.exp(np.clip(x, -5, 5))
        }

        G = {k: v for k, v in G_all.items() if allowed_functions is None or k in allowed_functions}

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

        g_Z_X_name = np.random.choice(list(G.keys()))
        g_U_X_name = np.random.choice(list(G.keys()))
        g_X_Y_name = np.random.choice(list(G.keys()))
        g_U_Y_name = np.random.choice(list(G.keys()))

        g_Z_X = G[g_Z_X_name]
        g_U_X = G[g_U_X_name]
        g_X_Y = G[g_X_Y_name]
        g_U_Y = G[g_U_Y_name]

        Z = np.random.normal(0, 1, n)
        U = np.random.normal(0, 1, n)

        epsilon_X = np.random.normal(0, sigma_X, n)
        X = b_Z_X * g_Z_X(Z) + b_U_X * g_U_X(U) + epsilon_X

        epsilon_Y = np.random.normal(0, sigma_Y, n)
        Y = b_X_Y * g_X_Y(X) + b_U_Y * g_U_Y(U) + epsilon_Y

        Y1 = b_X_Y * g_X_Y(X + 1) + b_U_Y * g_U_Y(U)
        Y0 = b_X_Y * g_X_Y(X - 1) + b_U_Y * g_U_Y(U)
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
            'ATE_true': ATE_true,
            'PNS_true': PNS_true,
            'p_Y1_mean': np.mean(Y1),
            'p_Y0_mean': np.mean(Y0),
            'Z': Z,
            'U': U,
            'X': X,
            'Y': Y,
        }
