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
        "ATE_2SLS_0.99": lambda self: self.bound_ate_2SLS(0.99),
        "ATE_2SLS_0.98": lambda self: self.bound_ate_2SLS(0.98),

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
                        
        "ATE_entropybounds_0.80": lambda self: self.bound_entropy(entr=0.80, query='ATE'),
        "ATE_entropybounds_0.20": lambda self: self.bound_entropy(entr=0.20, query='ATE'),
        "ATE_entropybounds_0.10": lambda self: self.bound_entropy(entr=0.10, query='ATE'),
        "PNS_entropybounds_0.80": lambda self: self.bound_entropy(entr=0.80, query='PNS'),
        "PNS_entropybounds_0.20": lambda self: self.bound_entropy(entr=0.20, query='PNS'),
        "PNS_entropybounds_0.10": lambda self: self.bound_entropy(entr=0.10, query='PNS'),

        "ATE_zaffalonbounds": lambda self: ZaffalonBounds.bound_binaryIV(self.data, "ATE"),
        "PNS_zaffalonbounds": lambda self: ZaffalonBounds.bound_binaryIV(self.data, "PNS"),

        "ATE_manski": lambda self: self.bound_manski('ATE'),
        "PNS_manski": lambda self: self.bound_manski('PNS'),

    }

    def __init__(self, dag, dataframe):
        super().__init__(dag)
        self.data = dataframe


    def run_all_bounding_algorithms(self, algorithms=None):
        """
        Run all bounding algorithms, print runtime statistics, and return the runtimes and current timestamp.

        Args:
            algorithms (list, optional): List of algorithm names to run. If None, all algorithms are run.

        Returns:
            dict: A dictionary with algorithm runtimes and the current timestamp.
                  Example: {"runtimes": {"2SLS": 1.23, "manski": 0.45}, "timestamp": "2023-03-15T12:34:56"}
        """
        available_algorithms = self.AVAILABLE_ALGORITHMS

        if algorithms is None:
            algorithms = available_algorithms.keys()

        runtimes = {}
        total_start_time = time.time()

        for algo in algorithms:
            if algo in available_algorithms:
                print(f"Running {algo}...")
                start_time = time.time()
                available_algorithms[algo](self)
                end_time = time.time()
                runtime = end_time - start_time
                runtimes[algo] = runtime
                print(f"{algo} completed in {runtime:.2f} seconds.")
            else:
                print(f"Algorithm '{algo}' is not recognized.")

        total_end_time = time.time()
        total_runtime = total_end_time - total_start_time
        print(f"Total runtime: {total_runtime:.2f} seconds.")

        current_timestamp = datetime.now().isoformat()
        return {"runtimes": runtimes, "timestamp": current_timestamp}


    def bound_manski(self, query):
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
                    slack = 1 - p1 - p0
                    lower = p1 - p0 - slack
                    upper = p1 - p0 + slack
                    lower = max(lower, -1)
                    upper = min(upper, 1)

                elif query == 'PNS':
                    lower = max(0, p1 - p0)
                    upper = min(p1, 1 - p0)

                # Ensure logical ordering
                lower, upper = min(lower, upper), max(lower, upper)

            except Exception:
                lower = 0 if query == 'PNS' else -1
                upper = 1
                failed = True

            # Validity check only makes sense for ATE (if ATE_true is in the data)
            if query == 'ATE':
                bounds_valid = lower <= sim['ATE_true'] <= upper
            else:
                bounds_valid = lower <= sim['PNS_true'] <= upper

            bounds_width = upper - lower

            self.data.at[idx, f'{query}_manski_bound_lower'] = lower
            self.data.at[idx, f'{query}_manski_bound_upper'] = upper
            self.data.at[idx, f'{query}_manski_bound_width'] = bounds_width
            self.data.at[idx, f'{query}_manski_bound_failed'] = failed
            self.data.at[idx, f'{query}_manski_bound_valid'] = bounds_valid




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
            self.data.at[idx, 'ATE_2SLS_' + ci_level_str + '_bound_lower'] = CI_lower
            self.data.at[idx, 'ATE_2SLS_' + ci_level_str + '_bound_upper'] = CI_upper
            self.data.at[idx, 'ATE_2SLS_' + ci_level_str + '_bound_valid'] = CI_valid
            self.data.at[idx, 'ATE_2SLS_' + ci_level_str + '_bound_width'] = CI_width
            self.data.at[idx, 'ATE_2SLS_' + ci_level_str + '_bound_failed'] = failed
    

    @staticmethod
    def generate_data_rolling_ate(N_simulations=2000, n=500, seed=None, b_U_X=None, b_U_Y=None, b_Z=None, intercept_X=None, intercept_Y=None, p_U=None, p_Z=None):
        """
        Generate data for a binary instrumental variable scenario.

        Args:
            N_simulations (int): Number of simulations to run. Default is 2000.
            n (int): Number of samples to generate. Default is 500.
            seed (int, optional): Random seed for reproducibility. Default is None.
            b_U_X (float): Coefficient for the effect of unobserved confounder U on X. Default is drawn from N(0, 1).
            b_U_Y (float): Coefficient for the effect of unobserved confounder U on Y. Default is drawn from N(0, 1).
            b_Z (float): Coefficient for the effect of instrument Z on X. Default is drawn from a bimodal distribution.
            intercept_X (float): Intercept for the logistic model of X. Default is 0.
            intercept_Y (float): Intercept for the logistic model of Y. Default is 0.
            p_U (float): Probability of unobserved confounder U ~ Bernoulli(p_U). Default is drawn from Uniform(0, 1).
            p_Z (float): Probability of instrument Z ~ Bernoulli(p_Z). Default is drawn from Uniform(0, 1).

        Returns:
            dict: A dictionary containing the generated data and parameters.
        """
        df_results = []
        
        # determine step size such that we run N_simulations simulations
        # in the range of -5 to 5
        step_size = (5 - (-5)) / N_simulations

        for b_X_Y in np.arange(-5, 5, step_size):
            result = BinaryIV._simulate_deterministic_data(
                n=n,
                seed=seed,
                b_U_X=b_U_X,
                b_U_Y=b_U_Y,
                b_Z=b_Z,
                b_X_Y=b_X_Y,
                intercept_X=intercept_X,
                intercept_Y=intercept_Y,
                p_U=p_U,
                p_Z=p_Z
            )
            df_results.append(result)
        return pd.DataFrame(df_results)

    def _simulate_deterministic_data(
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
        Simulate deterministic (binary) data for causal analysis.

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
                - ATE_true (float): True Average Treatment Effect of X on Y.
                - PNS_true (float): True Probability of Necessity and Suffiency of X on Y.
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
        PNS_i = p_Y1 * (1 - p_Y0)
        PNS_true = np.mean(PNS_i)


        return {
            'seed': seed,
            'intercept_X': intercept_X,
            'intercept_Y': intercept_Y,
            'b_Z': b_Z,
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
            'entropy_Z': EntropyBounds.entropy_of_array(Z),
            'entropy_U': EntropyBounds.entropy_of_array(U),
            'entropy_X': EntropyBounds.entropy_of_array(X),
            'entropy_Y': EntropyBounds.entropy_of_array(Y)
            
        }

