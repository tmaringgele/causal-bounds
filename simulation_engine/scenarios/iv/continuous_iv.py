from ..scenario import Scenario
from .binary_iv import BinaryIV
import numpy as np
import pandas as pd
from simulation_engine.util.datagen_util import datagen_util
from scipy.stats import norm
from simulation_engine.algorithms.zhang_bareinboim import ZhangBareinboim
from linearmodels.iv import IV2SLS
import warnings



class ContinuousIV(Scenario):

    AVAILABLE_ALGORITHMS = {
        "ATE_zhangbareinboim": lambda self: ZhangBareinboim.bound_ATE(self.data),
        "ATE_causaloptim--binned": lambda self: self.run_binaryIV('ATE_causaloptim'),
        "ATE_autobound--binned": lambda self: self.run_binaryIV('ATE_autobound'),
        "ATE_zaffalonbounds--binned": lambda self: self.run_binaryIV('ATE_zaffalonbounds'),
        "ATE_2SLS-0.99": lambda self: self.bound_ate_2SLS(0.99),
        "ATE_2SLS-0.98": lambda self: self.bound_ate_2SLS(0.98),
        "ATE_2SLS-0.95": lambda self: self.bound_ate_2SLS(0.95),
        "ATE_entropybounds-0.80--binned": lambda self: self.run_binaryIV('ATE_entropybounds-0.80'),
        "ATE_entropybounds-0.20--binned": lambda self: self.run_binaryIV('ATE_entropybounds-0.20'),
        "ATE_entropybounds-0.10--binned": lambda self: self.run_binaryIV('ATE_entropybounds-0.10'),
    }

    def __init__(self, dag, dataframe_cont, cutoff=0.5):
        super().__init__(dag)
        self.cutoff = cutoff
        self.data = dataframe_cont
        binned_data = self._bin_data(dataframe_cont, cutoff)
        # Create internal binary IV object
        self.binaryIV = BinaryIV(dag, binned_data)

    def _bin_data(self, dataframe_cont, cutoff=0.5):
        data = dataframe_cont.copy()
        # Ensure data['Y'] is a 1D array or Series of scalars before binarization
        for idx, row in data.iterrows():
            y_array = np.array(row['Y'], copy=True)  # Make a copy to avoid modifying original
            for i in range(len(y_array)):
                y_array[i] = 1 if y_array[i] > cutoff else 0
            data.at[idx, 'Y'] = y_array.astype(int)
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
                new_col_name = f"{algorithm}--binned{col_without_name}"
                self.data[new_col_name] = self.binaryIV.data[col]

        print("Exiting binned Binary IV Scenario.")
        
        return self.data

    def bound_ate_2SLS(self, ci_level=0.98):
        """
        Compute 2SLS bounds for the ATE using the given confidence level.

        Args:
            ci_level (float): Confidence level for the bounds. Default is 0.98.

        Returns:
            Void: This method modifies the self.data DataFrame in place.
        """
        for idx, sim in self.data.iterrows():
            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            # Add a constant term for the exogenous variables
            df['const'] = 1  # Adding a constant column

            # Define the dependent variable (Y), endogenous variable (X), exogenous variable (constant), and instrument (Z)
            dependent = df['Y']
            endog = df['X']
            exog = df[['const']]  # Exogenous variables (constant term)
            instruments = df['Z']

            failed = False
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # Perform 2SLS regression
                    model = IV2SLS(dependent, exog, endog, instruments).fit()
                    CI_upper = model.conf_int(level=ci_level).loc['X']['upper']
                    if CI_upper > 1:
                        CI_upper = 1

                    CI_lower = model.conf_int(level=ci_level).loc['X']['lower']
                    if CI_lower < -1:
                        CI_lower = -1
                    # If any warnings were raised, treat as failure
                    if len(w) > 0:
                        raise RuntimeError(f"2SLS produced warnings: {[str(warn.message) for warn in w]}")
            except Exception as e:
                # print(f"2SLS failed for simulation {idx} with error: {e}")
                CI_upper = 1
                CI_lower = -1
                failed = True

            CI_valid = CI_lower <= sim['ATE_true'] <= CI_upper
            CI_width = CI_upper - CI_lower

            ci_level_str = f"{ci_level:.2f}"
            self.data.at[idx, 'ATE_2SLS-' + ci_level_str + '_bound_lower'] = CI_lower
            self.data.at[idx, 'ATE_2SLS-' + ci_level_str + '_bound_upper'] = CI_upper
            self.data.at[idx, 'ATE_2SLS-' + ci_level_str + '_bound_valid'] = CI_valid
            self.data.at[idx, 'ATE_2SLS-' + ci_level_str + '_bound_width'] = CI_width
            self.data.at[idx, 'ATE_2SLS-' + ci_level_str + '_bound_failed'] = failed




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
                    allowed_functions=allowed_functions  # <-- pass through
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
        allowed_functions=None,
        intercept_X=None,
        intercept_Y=None
    ):
        if seed is None:
            seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        # Structural parameters
        if sigma_X is None:
            sigma_X = np.abs(np.random.normal(0, 1))
        if sigma_Y is None:
            sigma_Y = np.abs(np.random.normal(0, 1))
        if b_Z_X is None:
            b_Z_X = datagen_util.pick_from_bimodal()
        if b_U_X is None:
            b_U_X = datagen_util.pick_from_bimodal()
        if b_X_Y is None:
            b_X_Y = datagen_util.pick_from_bimodal()
        if b_U_Y is None:
            b_U_Y = datagen_util.pick_from_bimodal()
        if p_Z is None:
            p_Z = np.random.uniform(0, 1)
        if sigma_U is None:
            sigma_U = np.abs(np.random.normal(0, 1))
        if intercept_X is None:
            intercept_X = np.random.normal(0, 1)
        if intercept_Y is None:
            intercept_Y = np.random.normal(0, 1)

        # Nonlinear functions
        G_all = {
            "identity": lambda x: x,
            "sin": np.sin,
            "cos": np.cos,
            "tanh": np.tanh,
            "log1p_abs": lambda x: np.log1p(np.abs(x)),
            "exp_neg_sq": lambda x: np.exp(-x**2),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "exp_clipped": lambda x: np.exp(np.clip(x, -5, 5)),
            "zero_centered_tanh": lambda x: np.tanh(x),
            "sigmoid_shifted": lambda x: 1 / (1 + np.exp(-(x - np.mean(x)))),
            "sine_sym": lambda x: np.sin(x * np.pi),
            "bounded_linear": lambda x: np.clip(x / 5, -1, 1),
            "rescaled_identity": lambda x: x / (1 + np.abs(x)),
        }

        if allowed_functions is not None:
            G_all = {k: v for k, v in G_all.items() if k in allowed_functions}

        # Exogenous variables
        Z = np.random.binomial(1, p_Z, n)
        U = np.random.normal(0, sigma_U, n)  # Homoskedastic U for now

        # Heteroskedastic noise for X: σ_i ∼ |N(0, σ_X)|
        sigma_X_vec = np.abs(np.random.normal(0, sigma_X, size=n))
        epsilon_X = np.random.normal(0, sigma_X_vec)

        g_U_X_name = np.random.choice(list(G_all.keys()))
        g_U_X = G_all[g_U_X_name]
        X_latent = intercept_X + b_Z_X * Z + b_U_X * g_U_X(U) + epsilon_X

        squashers = datagen_util.get_squashers()
        squash_name = np.random.choice(list(squashers.keys()))
        squasher = squashers[squash_name]
        p_X = squasher(X_latent)
        X = np.random.binomial(1, p_X)

        # Heteroskedastic noise for Y: σ_i ∼ |N(0, σ_Y)|
        sigma_Y_vec = np.abs(np.random.normal(0, sigma_Y, size=n))
        epsilon_Y = np.random.normal(0, sigma_Y_vec)


        g_U_Y_name = np.random.choice(list(G_all.keys()))
        g_U_Y = G_all[g_U_Y_name]

        Y_raw = intercept_Y + b_X_Y * X + b_U_Y * g_U_Y(U) + epsilon_Y

        # Squash and clip Y
        squash_Y_name = np.random.choice(list(squashers.keys()))
        squasher_Y = squashers[squash_Y_name]
        Y = squasher_Y(Y_raw)
        Y = np.clip(Y, 0, 1)

        # Counterfactual outcomes using the *same noise* (factual counterfactual pair assumption)
        Y1_raw = b_X_Y * 1 + b_U_Y * g_U_Y(U) + epsilon_Y
        Y0_raw = b_X_Y * 0 + b_U_Y * g_U_Y(U) + epsilon_Y
        Y1 = squasher_Y(Y1_raw)
        Y0 = squasher_Y(Y0_raw)
        Y1 = np.clip(Y1, 0, 1)
        Y0 = np.clip(Y0, 0, 1)

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
            'intercept_X': intercept_X,
            'intercept_Y': intercept_Y,
            'g_U_X': g_U_X_name,
            'g_U_Y': g_U_Y_name,
            'squash_X': squash_name,
            'squash_Y': squash_Y_name,
            'ATE_true': ATE_true,
            'PNS_true': PNS_true,
            'p_Y1_mean': np.mean(Y1),
            'p_Y0_mean': np.mean(Y0),
            'Z': Z,
            'U': U,
            'X': X,
            'Y': Y,
            'epsilon_X': epsilon_X,
            'epsilon_Y': epsilon_Y,
            'sigma_X_vec': sigma_X_vec,
            'sigma_Y_vec': sigma_Y_vec,
            'Y_max': np.max(Y),
            'Y_min': np.min(Y),
            'X_max': np.max(X),
            'X_min': np.min(X),
            'X_mean': np.mean(X),
            'Y_mean': np.mean(Y),
            'entropy_Z': datagen_util.safe_entropy(Z),
            'entropy_X': datagen_util.safe_entropy(X),
        }


