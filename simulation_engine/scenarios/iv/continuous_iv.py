from .base_iv import IVScenario
from .binary_iv import BinaryIV
import numpy as np
import pandas as pd
from simulation_engine.util.datagen_util import datagen_util
from scipy.stats import norm
from simulation_engine.algorithms.zhang_bareinboim import ZhangBareinboim


class ContinuousIV(IVScenario):

    AVAILABLE_ALGORITHMS = {
        "ATE_zhangbareinboim": lambda self: ZhangBareinboim.bound_ATE(self.data),
        "ATE_causaloptim-binned": lambda self: self.run_binaryIV('ATE_causaloptim'),
        "ATE_autobound-binned": lambda self: self.run_binaryIV('ATE_autobound'),
    }

    def __init__(self, dag, dataframe_cont, cutoff=0.5):
        super().__init__(dag)
        self.cutoff = cutoff
        self.data = dataframe_cont
        binned_data = self._bin_data(dataframe_cont)
        # Create internal binary IV object
        self.binaryIV = BinaryIV(dag, binned_data)

    def _bin_data(self, dataframe_cont, cutoff=0.5):
        data = dataframe_cont.copy()
        # Ensure data['Y'] is a 1D array or Series of scalars before binarization
        for idx, row in data.iterrows():
            y_array = row['Y']
            #for each element of y_array
            for i in range(len(y_array)):
                #set to 1 if greater than cutoff, else 0
                y_array[i] = 1 if y_array[i] > cutoff else 0
            data.at[idx, 'Y'] = y_array
            
        return data  
    
    def run_binaryIV(self, algorithm='ATE_causaloptim'):
        """
        Run the binary IV algorithm on the continuous IV data.

        Args:
            algorithm (str): The name of the algorithm to run. Default is 'ATE_causaloptim'.

        Returns:
            pd.DataFrame: The results of the binary IV algorithm.
        """
        if algorithm not in self.binaryIV.get_algorithms():
            raise ValueError(f"Algorithm '{algorithm}' is not available.")
        
        print("Entering binned Binary IV Scenario for algorithm:", algorithm)
        # Run the specified algorithm on the binary IV object
        self.binaryIV.run([algorithm])

        # add the result cols to the continuous IV data
        # they all start with the algorithm name
        for col in self.binaryIV.data.columns:
            if col.startswith(algorithm):
                #add '-binned-0.5' to the algorithm name
                col_without_name = col.replace(algorithm, '')
                new_col_name = f"{algorithm}-binned{col_without_name}"
                self.data[new_col_name] = self.binaryIV.data[col]

        print("Exiting binned Binary IV Scenario.")
        
        return self.data


    


    
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
        seed=None):
        """
        Run simulations across a range of b_X_Y values with multiple replications per point,
        returning a DataFrame with all individual simulation results (not aggregated).

        Args:
            b_range (tuple): Range (min, max) of b_X_Y values.
            N_points (int): Number of evenly spaced b_X_Y values to try.
            replications (int): Number of replications per b_X_Y value.
            n (int): Sample size per simulation.
            seed (int or None): Optional base seed.

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
                    b_X_Y=b_X_Y_val
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
        p_Z=None,
        sigma_U=None,
    ):
        if seed is None:
            seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        # Structural coefficients
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
        if p_Z is None:
            p_Z = np.random.uniform(0, 1)
        if sigma_U is None:
            sigma_U = np.abs(np.random.normal(0, 1))

        # Cool nonlinear functions
        G_all = {
            "identity": lambda x: x,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "log1p_abs": lambda x: np.log1p(np.abs(x)),  # Positives only — tends high
            "exp_neg_sq": lambda x: np.exp(-x**2),       # Always (0,1), peaked at 0 — good
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "exp_clipped": lambda x: np.exp(np.clip(x, -5, 5)),

            # New additions to center things:
            "zero_centered_tanh": lambda x: np.tanh(x),  # symmetric, maps to (-1, 1)
            "sigmoid_shifted": lambda x: 1 / (1 + np.exp(-(x - np.mean(x)))),  # normalized input
            "sine_sym": lambda x: np.sin(x * np.pi),  # ensures output is in [-1, 1] periodically
            "bounded_linear": lambda x: np.clip(x / 5, -1, 1),  # linear but bounded
            "rescaled_identity": lambda x: x / (1 + np.abs(x)),  # smooth contraction
        }

        # Binary instrument and unobserved confounder
        Z = np.random.binomial(1, p_Z, n)
        U = np.random.normal(0, sigma_U, n)

        # Latent treatment score and stochastic binarization
        g_U_X_name = np.random.choice(list(G_all.keys()))
        g_U_X = G_all[g_U_X_name]
        X_latent = b_Z_X * Z + b_U_X * g_U_X(U) + np.random.normal(0, sigma_X, n)

        squashers = datagen_util.get_squashers()
        squash_name = np.random.choice(list(squashers.keys()))
        squasher = squashers[squash_name]
        p_X = squasher(X_latent)
        X = np.random.binomial(1, p_X)


        g_Y_name = np.random.choice(list(G_all.keys()))
        g_Y = G_all[g_Y_name]

        # Outcome Y
        g_U_Y_name = np.random.choice(list(G_all.keys()))
        g_U_Y = G_all[g_U_Y_name]
        epsilon_Y = np.random.normal(0, sigma_Y, n)
        Y_raw = b_X_Y * X + b_U_Y * g_U_Y(U) + epsilon_Y

        Y_clips = {
            'lower': 0,
            'upper': 1
        }

        Y = np.clip(g_Y(Y_raw), Y_clips['lower'], Y_clips['upper'])

        # Counterfactual outcomes
        eps1 = np.random.normal(0, sigma_Y, n)
        eps0 = np.random.normal(0, sigma_Y, n)
        Y1_raw = b_X_Y * 1 + b_U_Y * U + eps1
        Y0_raw = b_X_Y * 0 + b_U_Y * U + eps0
        Y1 = np.clip(g_Y(Y1_raw), Y_clips['lower'], Y_clips['upper'])
        Y0 = np.clip(g_Y(Y0_raw), Y_clips['lower'], Y_clips['upper'])

        ATE_true = np.mean(Y1 - Y0)
        PNS_true = np.mean((Y1 > Y0).astype(float))

        return {
            'seed': seed,
            'b_Z_X': b_Z_X,
            'b_U_X': b_U_X,
            'b_X_Y': b_X_Y,
            'b_U_Y': b_U_Y,
            'p_Z': p_Z,
            'sigma_X': sigma_X,
            'sigma_Y': sigma_Y,
            'sigma_U': sigma_U,
            'g_U_X': g_U_X_name,
            'g_U_Y': g_U_Y_name,
            'squash_X': squash_name,
            'g_Y': g_Y_name,
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
