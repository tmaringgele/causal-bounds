from simulation_engine.util.alg_util import AlgUtil
from .base_iv import IVScenario
from simulation_engine.util.datagen_util import datagen_util
from simulation_engine.algorithms.causaloptim import Causaloptim
from simulation_engine.algorithms.autobound import AutoBound
from simulation_engine.algorithms.entropybounds import EntropyBounds
from simulation_engine.algorithms.zaffalonbounds import ZaffalonBounds
from linearmodels.iv import IV2SLS
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr

import time
from datetime import datetime


class BinaryIV(IVScenario):
    AVAILABLE_ALGORITHMS = {
        "ATE_2SLS-0.99": lambda self: self.bound_ate_2SLS(0.99),
        "ATE_2SLS-0.98": lambda self: self.bound_ate_2SLS(0.98),

        "ATE_causaloptim": lambda self: Causaloptim.bound("ATE", self.data, 
                       graph_str="(Z -+ X, X -+ Y, Ur -+ X, Ur -+ Y)", 
                       leftside=[1, 0, 0, 0], 
                       latent=[0, 0, 0, 1], 
                       nvals=[2, 2, 2, 2], 
                       rlconnect=[0, 0, 0, 0], 
                       monotone=[0, 0, 0, 0]),
        "PNS_causaloptim": lambda self: Causaloptim.bound("PNS", self.data,
                       graph_str="(Z -+ X, X -+ Y, Ur -+ X, Ur -+ Y)", 
                       leftside=[1, 0, 0, 0], 
                       latent=[0, 0, 0, 1], 
                       nvals=[2, 2, 2, 2], 
                       rlconnect=[0, 0, 0, 0], 
                       monotone=[0, 0, 0, 0]),

        "ATE_autobound": lambda self: AutoBound.bound_binaryIV("ATE", self.data, 
                        dagstring="Z -> X, X -> Y, U -> X, U -> Y",
                        unob="U",
                        ),
        "PNS_autobound": lambda self: AutoBound.bound_binaryIV("PNS", self.data, 
                        dagstring="Z -> X, X -> Y, U -> X, U -> Y",
                        unob="U",
                        ),
                        
        "ATE_entropybounds-0.80": lambda self: EntropyBounds.bound(self.data, 0.80, 'ATE'),
        "ATE_entropybounds-0.20": lambda self: EntropyBounds.bound(self.data, 0.20, 'ATE'),
        "ATE_entropybounds-0.10": lambda self: EntropyBounds.bound(self.data, 0.10, 'ATE'),
        "PNS_entropybounds-0.80": lambda self: EntropyBounds.bound(self.data, 0.80, 'PNS'),
        "PNS_entropybounds-0.20": lambda self: EntropyBounds.bound(self.data, 0.20, 'PNS'),
        "PNS_entropybounds-0.10": lambda self: EntropyBounds.bound(self.data, 0.10, 'PNS'),

        "ATE_zaffalonbounds": lambda self: ZaffalonBounds.bound_binaryIV(self.data, "ATE"),
        "PNS_zaffalonbounds": lambda self: ZaffalonBounds.bound_binaryIV(self.data, "PNS"),

        "ATE_nonpara": lambda self: self.bound_nonpara('ATE'),
        "PNS_nonpara": lambda self: self.bound_nonpara('PNS'),

    }

    def __init__(self, dag, dataframe):
        super().__init__(dag)
        self.data = dataframe





    def bound_nonpara(self, query):
        """
        Compute Manski-style bounds for a given query using only observed treatment (X) and outcome (Y).
        
        Supported queries:
            - 'ATE' : Average Treatment Effect
            - 'PNS' : Probability of Necessity and Sufficiency

        Args:
            query (str): One of 'ATE' or 'PNS'

        Returns:
            Void: Modifies self.data DataFrame in place by adding bound columns.
        """
        assert query in {'ATE', 'PNS'}, "Query must be either 'ATE' or 'PNS'"

        for idx, sim in self.data.iterrows():
            X = np.array(sim['X'])
            Y = np.array(sim['Y'])
            failed = False

            try:
                p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.0
                p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.0

                if query == 'ATE':
                    lower = p1 - p0 - 1
                    upper = p1 - p0 + 1
                    lower = max(lower, -1)
                    upper = min(upper, 1)

                elif query == 'PNS':
                    lower = max(0, p1 - p0)
                    upper = min(p1, 1 - p0)

                # Ensure logical ordering
                lower, upper = min(lower, upper), max(lower, upper)

            except Exception:
                failed = True

            # Flatten bounds to trivial ceils
            AlgUtil.flatten_bounds_to_trivial_ceils(query, lower, upper, failed)

            # Validity check only makes sense for ATE (if ATE_true is in the data)
            if query == 'ATE':
                bounds_valid = lower <= sim['ATE_true'] <= upper
            else:
                bounds_valid = lower <= sim['PNS_true'] <= upper

            bounds_width = upper - lower

            self.data.at[idx, f'{query}_nonpara_bound_lower'] = lower
            self.data.at[idx, f'{query}_nonpara_bound_upper'] = upper
            self.data.at[idx, f'{query}_nonpara_bound_width'] = bounds_width
            self.data.at[idx, f'{query}_nonpara_bound_failed'] = failed
            self.data.at[idx, f'{query}_nonpara_bound_valid'] = bounds_valid




    def bound_entropy(self, entr=0.5, query='ATE', method="cf",  randomize_theta=False):
        """
        Compute entropy bounds for the ATE using the given method and entropy constraint.

        Args:
            method (str): Method to use for computing bounds. Options are 'cf' or 'cp'. Default is 'cf'.
            entr (float): Upper bound on confounder entropy. Default is 0.5.
            query (str): The query type (e.g., 'ATE' or 'PNS') to compute bounds for.

        Returns:
            Void: This method modifies the self.data DataFrame in place.
        """
        assert method in {'cf', 'cp'}, "Method must be either 'cf' or 'cp'"
        assert query in {'ATE', 'PNS'}, "Query must be either 'ATE' or 'PNS'"

        for idx, sim in self.data.iterrows():
            if randomize_theta:
                # Randomize theta
                theta = np.random.uniform(0, 1)
            else:
                theta = entr

            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            failed = False
            
            # try:
            bound_lower, bound_upper = EntropyBounds.run_experiment_binaryIV(df, theta, query, method)
                
            # except Exception as e:
            #     failed = True
            
            #Flatten bounds to trivial ceils
            if failed | (bound_upper > AlgUtil.get_trivial_Ceils(query)[1]):
                bound_upper = AlgUtil.get_trivial_Ceils(query)[1] 
            if failed | (bound_lower < AlgUtil.get_trivial_Ceils(query)[0]): 
                bound_lower = AlgUtil.get_trivial_Ceils(query)[0]


            bounds_valid = bound_lower <= sim[query+'_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            theta_rounded = f"{theta:.2f}"  # Always show two decimal places, e.g., 0.50
            self.data.at[idx, f"{query}_entropybounds_{theta_rounded}_bound_lower"] = bound_lower
            self.data.at[idx, f"{query}_entropybounds_{theta_rounded}_bound_upper"] = bound_upper
            self.data.at[idx, f"{query}_entropybounds_{theta_rounded}_bound_valid"] = bounds_valid
            self.data.at[idx, f"{query}_entropybounds_{theta_rounded}_bound_width"] = bounds_width
            self.data.at[idx, f"{query}_entropybounds_{theta_rounded}_bound_failed"] = failed



        

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
                # Perform 2SLS regression
                model = IV2SLS(dependent, exog, endog, instruments).fit()
                CI_upper = model.conf_int(level=ci_level).loc['X']['upper']
                if CI_upper > 1:
                    CI_upper = 1

                CI_lower = model.conf_int(level=ci_level).loc['X']['lower']
                if CI_lower < -1:
                    CI_lower = -1
            except Exception as e:
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
    def generate_data_rolling_ate(N_simulations=2000, n=500, seed=None, b_U_X=None, b_U_Y=None, b_Z_X=None, intercept_X=None, intercept_Y=None, p_U=None, p_Z=None, sigma_X=None, sigma_Y=None):
        """
        Generate data for a binary instrumental variable scenario.

        Args:
            N_simulations (int): Number of simulations to run. Default is 2000.
            n (int): Number of samples to generate. Default is 500.
            seed (int, optional): Random seed for reproducibility. Default is None.
            b_U_X (float): Coefficient for the effect of unobserved confounder U on X. Default is drawn from N(0, 1).
            b_U_Y (float): Coefficient for the effect of unobserved confounder U on Y. Default is drawn from N(0, 1).
            b_Z_X (float): Coefficient for the effect of instrument Z on X. Default is drawn from a bimodal distribution.
            intercept_X (float): Intercept for the logistic model of X. Default is 0.
            intercept_Y (float): Intercept for the logistic model of Y. Default is 0.
            p_U (float): Probability of unobserved confounder U ~ Bernoulli(p_U). Default is drawn from Uniform(0, 1).
            p_Z (float): Probability of instrument Z ~ Bernoulli(p_Z). Default is drawn from Uniform(0, 1).
            sigma_X (float or None): Std. dev. of noise added to the X logit. If None, sampled from |N(0,1)|. If 0, no noise.
            sigma_Y (float or None): Std. dev. of noise added to the Y logit. Same behavior as sigma_X.

        Returns:
            dict: A dictionary containing the generated data and parameters.
        """
        df_results = []
        
        # determine step size such that we run N_simulations simulations
        # in the range of -5 to 5
        step_size = (5 - (-5)) / N_simulations

        for b_X_Y in np.linspace(-5, 5, N_simulations):
            result = BinaryIV._simulate_deterministic_data(
                n=n,
                seed=seed,
                b_U_X=b_U_X,
                b_U_Y=b_U_Y,
                b_Z_X=b_Z_X,
                b_X_Y=b_X_Y,
                intercept_X=intercept_X,
                intercept_Y=intercept_Y,
                p_U=p_U,
                p_Z=p_Z,
                sigma_X=sigma_X,
                sigma_Y=sigma_Y
            )
            df_results.append(result)
        return pd.DataFrame(df_results)

    def _simulate_deterministic_data(
        n=500,
        seed=None,
        b_U_X=None,
        b_U_Y=None,
        b_Z_X=None,
        b_X_Y=None,
        intercept_X=None,
        intercept_Y=None,
        p_U=None,
        p_Z=None,
        sigma_X=None,
        sigma_Y=None
    ):
        """
        Simulate binary data for causal analysis using a structural causal model with optional additive noise.
        Independent squashing functions are chosen for X and Y generation.

        Args:
            n (int): Number of samples to generate. Default is 500.
            seed (int, optional): Random seed for reproducibility. Default is None.
            b_U_X (float): Coefficient for the effect of U on X. Default is drawn from N(0, 1).
            b_U_Y (float): Coefficient for the effect of U on Y. Default is drawn from N(0, 1).
            b_Z_X (float): Coefficient for the effect of Z on X. Default is drawn from a bimodal distribution.
            b_X_Y (float): Coefficient for the effect of X on Y. Default is drawn from a bimodal distribution.
            intercept_X (float): Intercept for the logistic model of X. Default is 0.
            intercept_Y (float): Intercept for the logistic model of Y. Default is 0.
            p_U (float): Probability for U ~ Bernoulli(p_U). Default is drawn from Uniform(0, 1).
            p_Z (float): Probability for Z ~ Bernoulli(p_Z). Default is drawn from Uniform(0, 1).
            sigma_X (float or None): Std. dev. of noise added to the X logit. If None, sampled from |N(0,1)|. If 0, no noise.
            sigma_Y (float or None): Std. dev. of noise added to the Y logit. Same behavior as sigma_X.

        Returns:
            dict: Dictionary with all simulated data, model parameters, true ATE and PNS, and entropy values.
        """
        if seed is None:
            seed = np.random.randint(0, int(1e6))
        np.random.seed(seed)

        # Randomly choose separate squashing functions for X and Y
        squashers = datagen_util.get_squashers()
        squasher_X_name = np.random.choice(list(squashers.keys()))
        squasher_Y_name = np.random.choice(list(squashers.keys()))
        squasher_X = squashers[squasher_X_name]
        squasher_Y = squashers[squasher_Y_name]

        # Coefficient defaults
        if b_U_X is None:
            b_U_X = np.random.normal(0, 1)
        if b_U_Y is None:
            b_U_Y = np.random.normal(0, 1)
        if b_Z_X is None:
            b_Z_X = datagen_util.pick_from_bimodal()
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

        # Noise level logic
        if sigma_X is None:
            sigma_X = abs(np.random.normal(0, 1))
        if sigma_Y is None:
            sigma_Y = abs(np.random.normal(0, 1))

        # Binary exogenous variables
        Z = np.random.binomial(1, p_Z, size=n)
        U = np.random.binomial(1, p_U, size=n)

        # Add noise to treatment assignment
        epsilon_X = np.zeros(n) if sigma_X == 0 else np.random.normal(0, sigma_X, size=n)
        logit_X = intercept_X + b_Z_X * Z + b_U_X * U + epsilon_X
        p_X = squasher_X(logit_X)
        X = np.random.binomial(1, p_X)

        # Add noise to outcome assignment
        epsilon_Y = np.zeros(n) if sigma_Y == 0 else np.random.normal(0, sigma_Y, size=n)
        logit_Y = intercept_Y + b_X_Y * X + b_U_Y * U + epsilon_Y
        p_Y = squasher_Y(logit_Y)
        Y = np.random.binomial(1, p_Y)

        # Potential outcomes without noise
        logit_Y1 = intercept_Y + b_X_Y * 1 + b_U_Y * U
        logit_Y0 = intercept_Y + b_X_Y * 0 + b_U_Y * U
        p_Y1 = squasher_Y(logit_Y1)
        p_Y0 = squasher_Y(logit_Y0)
        ATE_true = np.mean(p_Y1 - p_Y0)
        PNS_true = np.mean(p_Y1 * (1 - p_Y0))

        return {
            'seed': seed,
            'intercept_X': intercept_X,
            'intercept_Y': intercept_Y,
            'b_Z_X': b_Z_X,
            'b_U_X': b_U_X,
            'b_X_Y': b_X_Y,
            'b_U_Y': b_U_Y,
            'ATE_true': ATE_true,
            'PNS_true': PNS_true,
            'p_Y1_mean': p_Y1.mean(),
            'p_Y0_mean': p_Y0.mean(),
            'p_U': p_U,
            'p_Z': p_Z,
            'Z': Z,
            'U': U,
            'X': X,
            'Y': Y,
            'entropy_Z': datagen_util.entropy_of_array(Z),
            'entropy_U': datagen_util.entropy_of_array(U),
            'entropy_X': datagen_util.entropy_of_array(X),
            'entropy_Y': datagen_util.entropy_of_array(Y),
            'sigma_X': sigma_X,
            'sigma_Y': sigma_Y,
            'squasher_X_name': squasher_X_name,
            'squasher_Y_name': squasher_Y_name
        }
    
    # Convert float arrays to int64 for entropy calculation
    def safe_entropy(arr):
        arr = np.asarray(arr).astype(np.int64)
        return datagen_util.entropy_of_array(arr)

    def generate_response_type_data(
        N_simulations=50,
        n=500,
        seed=None
    ):
        """
        Generate data for a binary instrumental variable scenario with response types.
        Did not really lead to meaningful results with the algorithms.
        Very High Fail Rates and often uninformative.
        """
        if seed is not None:
            np.random.seed(seed)

        response_types_Y = [
            (0, 0),  # always-no
            (0, 1),  # treatment helps
            (1, 0),  # treatment hurts
            (1, 1),  # always-yes
        ]

        x_response = (0, 1)  # perfect compliers

        results = []

        for help_frac in np.linspace(0, 1, N_simulations):
            remain = 1 - help_frac
            dist_Y = np.array([remain / 3, help_frac, remain / 3, remain / 3])

            Z = np.random.binomial(1, 0.5, size=n)
            U = np.random.binomial(1, 0.5, size=n)

            Y0 = np.zeros(n, dtype=np.int64)
            Y1 = np.zeros(n, dtype=np.int64)
            X = np.zeros(n, dtype=np.int64)
            Y = np.zeros(n, dtype=np.int64)

            type_counts = {str(rt): 0 for rt in response_types_Y}

            for i in range(n):
                idx = np.random.choice(4, p=dist_Y)
                y_type = response_types_Y[idx]
                Y0[i], Y1[i] = y_type
                type_counts[str(y_type)] += 1

                X[i] = x_response[Z[i]]
                Y[i] = Y1[i] if X[i] == 1 else Y0[i]

            ATE_true = np.mean(Y1 - Y0)
            PNS_true = np.mean((Y1 == 1) & (Y0 == 0))

            result = {
                'seed': seed,
                'intercept_X': None,
                'intercept_Y': None,
                'b_Z_X': None,
                'b_U_X': None,
                'b_X_Y': None,
                'b_U_Y': None,
                'ATE_true': ATE_true,
                'PNS_true': PNS_true,
                'p_Y1_mean': np.mean(Y1),
                'p_Y0_mean': np.mean(Y0),
                'p_U': 0.5,
                'p_Z': 0.5,
                'Z': Z,
                'U': U,
                'X': X,
                'Y': Y,
                'entropy_Z': BinaryIV.safe_entropy(Z),
                'entropy_U': BinaryIV.safe_entropy(U),
                'entropy_X': BinaryIV.safe_entropy(X),
                'entropy_Y': BinaryIV.safe_entropy(Y),
                'help_frac': help_frac,
                'response_type_dist': dist_Y,
                'response_type_counts': type_counts
            }

            results.append(result)

        return pd.DataFrame(results)



