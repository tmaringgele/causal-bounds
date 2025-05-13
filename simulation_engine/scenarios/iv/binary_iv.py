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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time


class BinaryIV(IVScenario):
    def __init__(self, dag, dataframe):
        super().__init__(dag)
        self.data = dataframe


    def run_all_bounding_algorithms(self, algorithms=None):
        """
        Run all bounding algorithms and print runtime statistics.

        Args:
            algorithms (list, optional): List of algorithm names to run. If None, all algorithms are run.
        """
        available_algorithms = {
            "2SLS": self.bound_ate_2SLS,
            "causaloptim": self.bound_ate_causaloptim,
            "autobound": self.bound_ate_autobound,
            "entropy": lambda: self.bound_ate_entropy(entr=0.80),
            "zaffalon_parallel": self.bound_ate_zaffalon_parallel,
            "manski": self.bound_ate_manski,
        }

        if algorithms is None:
            algorithms = available_algorithms.keys()

        for algo in algorithms:
            if algo in available_algorithms:
                print(f"Running {algo}...")
                start_time = time.time()
                available_algorithms[algo]()
                end_time = time.time()
                print(f"{algo} completed in {end_time - start_time:.2f} seconds.")
            else:
                print(f"Algorithm '{algo}' is not recognized.")


    def bound_ate_manski(self):
        """
        Compute Manski bounds for the ATE using only observed treatment (X) and outcome (Y),
        with no assumptions on confounding or instruments.

        Returns:
            Void: This method modifies the self.data DataFrame in place.
        """
        for idx, sim in self.data.iterrows():
            X = np.array(sim['X'])
            Y = np.array(sim['Y'])
            failed = False

            try:
                # Estimate P(Y=1 | T=1) and P(Y=1 | T=0)
                p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.0
                p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.0

                # Slack term
                slack = 1 - p1 - p0

                # Correct Manski bounds
                lower = p1 - p0 - slack
                upper = p1 - p0 + slack

                # Clip bounds to [-1, 1]
                lower = max(lower, -1)
                upper = min(upper, 1)

                # Ensure logical ordering
                lower, upper = min(lower, upper), max(lower, upper)

            except Exception:
                lower = -1
                upper = 1
                failed = True

            bounds_valid = lower <= sim['ATE_true'] <= upper
            bounds_width = upper - lower

            self.data.at[idx, 'manski_bound_lower'] = lower
            self.data.at[idx, 'manski_bound_upper'] = upper
            self.data.at[idx, 'manski_bound_valid'] = bounds_valid
            self.data.at[idx, 'manski_bound_width'] = bounds_width
            self.data.at[idx, 'manski_bound_failed'] = failed

    @staticmethod
    def _run_zaffalon_from_row_dict(row_dict):
        import pandas as pd
        try:
            df = pd.DataFrame({'Y': row_dict['Y'], 'X': row_dict['X'], 'Z': row_dict['Z']})
            from your_module import ZaffalonBounds  # import locally to be safe
            ate_lower, ate_upper = ZaffalonBounds.run_experiment_binaryIV_ATE(df)

            ate_lower = max(ate_lower, -1)
            ate_upper = min(ate_upper, 1)

            bounds_valid = ate_lower <= row_dict['ATE_true'] <= ate_upper
            bounds_width = ate_upper - ate_lower
            failed = False

        except Exception:
            ate_lower, ate_upper = -1, 1
            bounds_valid = False
            bounds_width = 2
            failed = True

        return {
            'zaffalonbounds_bound_lower': ate_lower,
            'zaffalonbounds_bound_upper': ate_upper,
            'zaffalonbounds_bound_valid': bounds_valid,
            'zaffalonbounds_bound_width': bounds_width,
            'zaffalonbounds_bound_failed': failed
        }

    def bound_ate_zaffalon_parallel(self, max_workers=None):

        row_dicts = [row.to_dict() for _, row in self.data.iterrows()]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(BinaryIV._run_zaffalon_from_row_dict, row_dicts))

        results_df = pd.DataFrame(results)
        self.data = pd.concat([self.data.reset_index(drop=True), results_df], axis=1)


    def bound_ate_zaffalon(self):
        """
        Compute entropy bounds for the ATE using the given method and entropy constraint.

        Args:
            method (str): Method to use for computing bounds. Options are 'cf' or 'cp'. Default is 'cf'.
            entr (float): Upper bound on confounder entropy. Default is 0.5.

        Returns:
            Void: This method modifies the self.data DataFrame in place.
        """
        for idx, sim in self.data.iterrows():

            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            failed = False
            try:
                ate_lower, ate_upper = ZaffalonBounds.run_experiment_binaryIV_ATE(df)
                # Flatten bounds to [2, 2]
                if ate_upper > 1: 
                    ate_upper = 1
                if ate_lower < -1: 
                    ate_lower = -1
            except Exception as e:
                ate_lower = -1
                ate_upper = 1
                failed = True

            bounds_valid = ate_lower <= sim['ATE_true'] <= ate_upper
            bounds_width = ate_upper - ate_lower

            self.data.at[idx, 'zaffalonbounds_bound_lower'] = ate_lower
            self.data.at[idx, 'zaffalonbounds_bound_upper'] = ate_upper
            self.data.at[idx, 'zaffalonbounds_bound_valid'] = bounds_valid
            self.data.at[idx, 'zaffalonbounds_bound_width'] = bounds_width
            self.data.at[idx, 'zaffalonbounds_bound_failed'] = failed


    def bound_ate_entropy(self, method="cf", entr=0.5, randomize_theta=False):
        """
        Compute entropy bounds for the ATE using the given method and entropy constraint.

        Args:
            method (str): Method to use for computing bounds. Options are 'cf' or 'cp'. Default is 'cf'.
            entr (float): Upper bound on confounder entropy. Default is 0.5.

        Returns:
            Void: This method modifies the self.data DataFrame in place.
        """
        for idx, sim in self.data.iterrows():
            if randomize_theta:
                # Randomize theta
                theta = np.random.uniform(0, 1)
            else:
                theta = entr

            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            failed = False
            try:
                ate_lower, ate_upper = EntropyBounds.run_experiment_binaryIV_ATE(df, entr=theta, method=method)
                # Flatten bounds to [2, 2]
                if ate_upper > 1: 
                    ate_upper = 1
                if ate_lower < -1: 
                    ate_lower = -1
            except Exception as e:
                ate_lower = -1
                ate_upper = 1
                failed = True

            bounds_valid = ate_lower <= sim['ATE_true'] <= ate_upper
            bounds_width = ate_upper - ate_lower

            self.data.at[idx, 'entropybounds_bound_lower'] = ate_lower
            self.data.at[idx, 'entropybounds_bound_upper'] = ate_upper
            self.data.at[idx, 'entropybounds_bound_valid'] = bounds_valid
            self.data.at[idx, 'entropybounds_bound_width'] = bounds_width
            self.data.at[idx, 'entropybounds_bound_failed'] = failed
            self.data.at[idx, 'entropybounds_H(conf)_UB'] = theta

    def bound_ate_autobound(self):
        """
        Compute autobound bounds for the ATE using the given confidence level.
        Args:
            None: This method modifies the self.data DataFrame in place.
        Returns:
            Void: This method modifies the self.data DataFrame in place.
        """
        for idx, sim in self.data.iterrows():
            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            failed = False
            try:
                bound_lower, bound_upper = AutoBound.run_experiment_binaryIV_ATE(df)
                # Flatten bounds to [2, 2]
                if bound_upper > 1: 
                    bound_upper = 1
                if bound_lower < -1: 
                    bound_lower = -1
            except Exception as e:
                bound_lower = -1
                bound_upper = 1
                failed = True

            bounds_valid = bound_lower <= sim['ATE_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            self.data.at[idx, 'autobound_bound_lower'] = bound_lower
            self.data.at[idx, 'autobound_bound_upper'] = bound_upper
            self.data.at[idx, 'autobound_bound_valid'] = bounds_valid
            self.data.at[idx, 'autobound_bound_width'] = bounds_width
            self.data.at[idx, 'autobound_bound_failed'] = failed

        

    # enriches self.data with causaloptim bound information
    def bound_ate_causaloptim(self):
        # Define Scenario and target quantity
        graph_str = "(Z -+ X, X -+ Y, Ur -+ X, Ur -+ Y)"
        leftside = [1, 0, 0, 0]
        latent = [0, 0, 0, 1]
        nvals = [2, 2, 2, 2]
        rlconnect = [0, 0, 0, 0]
        monotone = [0, 0, 0, 0]

        importr('causaloptim')
        importr('base')

        for idx, sim in self.data.iterrows():
            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            failed = False
            # try:

            result = Causaloptim.run_experiment(graph_str, leftside, latent, nvals, rlconnect, monotone, df)
            bound_lower = result['lower_bound']
            bound_upper = result['upper_bound']
            failed = result['failed'] 
            #Flatten bounds to [2, 2]
            if bound_upper > 1: 
                bound_upper = 1
            if bound_lower < -1: 
                bound_lower = -1
            
            # except Exception as e:
            #     failed = True

            # if failed:
            #     bound_lower = -1
            #     bound_upper = 1


            bounds_valid = bound_lower <= sim['ATE_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            
            self.data.at[idx, 'causaloptim_bound_lower'] = bound_lower
            self.data.at[idx, 'causaloptim_bound_upper'] = bound_upper
            self.data.at[idx, 'causaloptim_bound_valid'] = bounds_valid
            self.data.at[idx, 'causaloptim_bound_width'] = bounds_width
            self.data.at[idx, 'causaloptim_bound_failed'] = failed


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

            self.data.at[idx, '2SLS_bound_lower'] = CI_lower
            self.data.at[idx, '2SLS_bound_upper'] = CI_upper
            self.data.at[idx, '2SLS_bound_valid'] = CI_valid
            self.data.at[idx, '2SLS_bound_width'] = CI_width
            self.data.at[idx, '2SLS_bound_failed'] = failed
    

    @staticmethod
    def generate_data_rolling_ate(N_simulations=2000, n=500, seed=None, b_U_X=None, b_U_Y=None, b_Z=None, intercept_X=None, intercept_Y=None):
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

        Returns:
            dict: A dictionary containing the generated data and parameters.
        """
        df_results = []
        
        # determine step size such that we run N_simulations simulations
        # in the range of -5 to 5
        step_size = (5 - (-5)) / N_simulations

        for b_X_Y in np.arange(-5, 5, step_size):
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
            'Y': Y,
            'entropy_Z': EntropyBounds.entropy_of_array(Z),
            'entropy_U': EntropyBounds.entropy_of_array(U),
            'entropy_X': EntropyBounds.entropy_of_array(X),
            'entropy_Y': EntropyBounds.entropy_of_array(Y)
            
        }

