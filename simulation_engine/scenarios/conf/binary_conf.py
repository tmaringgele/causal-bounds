import numpy as np
import pandas as pd
from simulation_engine.algorithms.autobound import AutoBound
from simulation_engine.algorithms.causaloptim import Causaloptim
from simulation_engine.algorithms.entropybounds import EntropyBounds
from simulation_engine.algorithms.manski import Manski
from simulation_engine.algorithms.tianpearl import TianPearl
from simulation_engine.scenarios.scenario import Scenario
from simulation_engine.util.datagen_util import datagen_util


class BinaryConf(Scenario):
    """
    Binary Confounder Scenario
    """

    def __init__(self, dag, dataframe):
        super().__init__(dag)
        self.data = dataframe

    AVAILABLE_ALGORITHMS = {
        "ATE_OLS-0.99": lambda self: self.bound_ate_OLS(0.99),
        "ATE_OLS-0.98": lambda self: self.bound_ate_OLS(0.98),
        "ATE_OLS-0.95": lambda self: self.bound_ate_OLS(0.95),

        "ATE_causaloptim": lambda self: Causaloptim.bound("ATE", self.data, 
                       graph_str="(X -+ Y, Ur -+ X, Ur -+ Y)", 
                       leftside=[0, 0, 0], 
                       latent=[0, 0, 1], 
                       nvals=[2, 2, 2], 
                       rlconnect=[0, 0, 0], 
                       monotone=[0, 0, 0]),
        "PNS_causaloptim": lambda self: Causaloptim.bound("PNS", self.data,
                       graph_str="(X -+ Y, Ur -+ X, Ur -+ Y)", 
                       leftside=[0, 0, 0], 
                       latent=[0, 0, 1], 
                       nvals=[2, 2, 2], 
                       rlconnect=[0, 0, 0], 
                       monotone=[0, 0, 0]),

        "ATE_autobound": lambda self: AutoBound.bound_binaryIV("ATE", self.data, 
                        dagstring="X -> Y, U -> X, U -> Y",
                        unob="U",
                        ),
        "PNS_autobound": lambda self: AutoBound.bound_binaryIV("PNS", self.data, 
                        dagstring="X -> Y, U -> X, U -> Y",
                        unob="U",
                        ),
                        
        "ATE_entropybounds-0.80": lambda self: EntropyBounds.bound(self.data, 0.80, 'ATE'),
        "ATE_entropybounds-0.20": lambda self: EntropyBounds.bound(self.data, 0.20, 'ATE'),
        "ATE_entropybounds-0.10": lambda self: EntropyBounds.bound(self.data, 0.10, 'ATE'),
        "PNS_entropybounds-0.80": lambda self: EntropyBounds.bound(self.data, 0.80, 'PNS'),
        "PNS_entropybounds-0.20": lambda self: EntropyBounds.bound(self.data, 0.20, 'PNS'),
        "PNS_entropybounds-0.10": lambda self: EntropyBounds.bound(self.data, 0.10, 'PNS'),

        "PNS_entropybounds-trueTheta": lambda self: EntropyBounds.bound(self.data, query='PNS', true_theta=True),
        "ATE_entropybounds-trueTheta": lambda self: EntropyBounds.bound(self.data, query='ATE', true_theta=True),
        
        "PNS_entropybounds-randomTheta": lambda self: EntropyBounds.bound(self.data, query='PNS', randomize_theta=True),
        "ATE_entropybounds-randomTheta": lambda self: EntropyBounds.bound(self.data, query='ATE', randomize_theta=True),

        # "ATE_zaffalonbounds": lambda self: ZaffalonBounds.bound_binaryIV(self.data, "ATE"),
        # "PNS_zaffalonbounds": lambda self: ZaffalonBounds.bound_binaryIV(self.data, "PNS"),

        "ATE_tianpearl": lambda self: TianPearl.bound(self.data, 'ATE'),
        "PNS_tianpearl": lambda self: TianPearl.bound(self.data, 'PNS'),

        "ATE_manski": lambda self: Manski.bound_ATE(self.data)

    }

    def bound_ate_OLS(self, ci_level=0.98):
        """
        Compute OLS bounds for the ATE using the given confidence level.

        Args:
            ci_level (float): Confidence level for the bounds. Default is 0.98.

        Returns:
            Void: This method modifies the self.data DataFrame in place.
        """
        for idx, sim in self.data.iterrows():
            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X']})
            # Add a constant term for the exogenous variables
            df['const'] = 1  # Adding a constant column

            # Define the dependent variable (Y), endogenous variable (X), exogenous variable (constant), and instrument (Z)
            dependent = df['Y']
            endog = df['X']
            exog = df[['const']]  # Exogenous variables (constant term)

            failed = False
            try:
                import statsmodels.api as sm
                import statsmodels.stats.api as sms
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    # Perform OLS regression
                    model = sm.OLS(dependent, exog.join(endog)).fit(cov_type='HC3')  # HC3 is robust to heteroskedasticity
                    # Get confidence interval for X coefficient
                    CI = model.conf_int(alpha=1-ci_level).loc['X']
                    CI_lower = CI[0]
                    CI_upper = CI[1]
                    if CI_upper > 1:
                        CI_upper = 1
                    if CI_lower < -1:
                        CI_lower = -1
                    # If any warnings were raised, treat as failure
                    if len(w) > 0:
                        raise RuntimeError(f"OLS produced warnings: {[str(warn.message) for warn in w]}")
            except Exception as e:
                # print(f"OLS failed for simulation {idx} with error: {e}")
                CI_upper = 1
                CI_lower = -1
                failed = True

            CI_valid = CI_lower <= sim['ATE_true'] <= CI_upper
            CI_width = CI_upper - CI_lower

            ci_level_str = f"{ci_level:.2f}"
            self.data.at[idx, 'ATE_OLS-' + ci_level_str + '_bound_lower'] = CI_lower
            self.data.at[idx, 'ATE_OLS-' + ci_level_str + '_bound_upper'] = CI_upper
            self.data.at[idx, 'ATE_OLS-' + ci_level_str + '_bound_valid'] = CI_valid
            self.data.at[idx, 'ATE_OLS-' + ci_level_str + '_bound_width'] = CI_width
            self.data.at[idx, 'ATE_OLS-' + ci_level_str + '_bound_failed'] = failed

   

    def generate_data_rolling_ate(N_simulations=2000, n=500, b_lower=-5, b_upper=5, seed=None, b_U_X=None, b_U_Y=None, intercept_X=None, intercept_Y=None, p_U=None):
        """
        Generate data for a binary instrumental variable scenario.

        Args:
            N_simulations (int): Number of simulations to run. Default is 2000.
            n (int): Number of samples to generate. Default is 500.
            seed (int, optional): Random seed for reproducibility. Default is None.
            b_U_X (float): Coefficient for the effect of unobserved confounder U on X. Default is drawn from N(0, 1).
            b_U_Y (float): Coefficient for the effect of unobserved confounder U on Y. Default is drawn from N(0, 1).
            intercept_X (float): Intercept for the logistic model of X. Default is 0.
            intercept_Y (float): Intercept for the logistic model of Y. Default is 0.
            p_U (float): Probability of unobserved confounder U ~ Bernoulli(p_U). Default is drawn from Uniform(0, 1).
        Returns:
            dict: A dictionary containing the generated data and parameters.
        """
        df_results = []
        
        # determine step size such that we run N_simulations simulations
        # in the range of -5 to 5
        step_size = (5 - (-5)) / N_simulations

        for b_X_Y in np.linspace(b_lower, b_upper, N_simulations):
            result = BinaryConf._generate_data(
                n=n,
                seed=seed,
                b_U_X=b_U_X,
                b_U_Y=b_U_Y,
                b_X_Y=b_X_Y,
                intercept_X=intercept_X,
                intercept_Y=intercept_Y,
                p_U=p_U
                )
            df_results.append(result)
        return pd.DataFrame(df_results)

    @staticmethod
    def _generate_data(
        n=500,
        seed=None,
        b_U_X=None,
        b_U_Y=None,
        b_X_Y=None,
        intercept_X=None,
        intercept_Y=None,
        p_U=None,
        uniform_confounder_entropy=False
    ):
        """
        Simulate binary data using a structural causal model (SCM) with additive, heteroskedastic noise.

        Each simulation draws a sample of size `n`. Binary exogenous variables U and Z influence
        treatment X and outcome Y through logistic models with heteroskedastic Gaussian noise.
        Each observation has its own standard deviation for noise, drawn from |N(0,1)|.
        Independent squashing functions (e.g., sigmoid, tanh) are randomly selected for X and Y.

        Args:
            n (int): Number of observations to generate. Default is 500.
            seed (int, optional): Random seed for reproducibility.
            b_U_X, b_U_Y, b_Z_X, b_X_Y (float): Coefficients in the structural equations. 
                                                If None, sampled from a bimodal distribution.
            intercept_X, intercept_Y (float): Intercepts in the logistic models. If None, drawn from N(0,1).
            p_U, p_Z (float): Probabilities for binary exogenous variables U and Z. 
                            If None, drawn from Uniform(0,1).
            uniform_confounder_entropy (bool): If True, samples `p_U` to induce uniform entropy in U.

        Returns:
            dict: Contains simulated variables (Z, U, X, Y), true ATE and PNS, parameters used, 
                entropies, heteroskedastic noise vectors, and metadata.
        """
        if seed is None:
            seed = np.random.randint(0, int(1e6))
        np.random.seed(seed)

        # Random squashing functions for X and Y
        squashers = datagen_util.get_squashers()
        squasher_X_name = np.random.choice(list(squashers.keys()))
        squasher_Y_name = np.random.choice(list(squashers.keys()))
        squasher_X = squashers[squasher_X_name]
        squasher_Y = squashers[squasher_Y_name]

        # Structural parameters
        if b_U_X is None:
            b_U_X = datagen_util.pick_from_bimodal()
        if b_U_Y is None:
            b_U_Y = datagen_util.pick_from_bimodal()
        if b_X_Y is None:
            b_X_Y = datagen_util.pick_from_bimodal()
        if intercept_X is None:
            intercept_X = np.random.normal(0, 1)
        if intercept_Y is None:
            intercept_Y = np.random.normal(0, 1)
        if uniform_confounder_entropy:
            p_U = datagen_util._sample_p_with_uniform_entropy()
        elif p_U is None:
            p_U = np.random.uniform(0, 1)

        # Exogenous binary variables
        U = np.random.binomial(1, p_U, size=n)

        # Heteroskedastic noise: sigma_i ~ |N(0,1)| for each observation
        sigma_X_vec = np.abs(np.random.normal(0, 1, size=n))
        epsilon_X = np.random.normal(0, sigma_X_vec)
        logit_X = intercept_X + b_U_X * U + epsilon_X
        p_X = squasher_X(logit_X)
        X = np.random.binomial(1, p_X)

        sigma_Y_vec = np.abs(np.random.normal(0, 1, size=n))
        epsilon_Y = np.random.normal(0, sigma_Y_vec)
        logit_Y = intercept_Y + b_X_Y * X + b_U_Y * U + epsilon_Y
        p_Y = squasher_Y(logit_Y)
        Y = np.random.binomial(1, p_Y)

        # Potential outcomes without noise
        logit_Y1 = intercept_Y + b_X_Y * 1 + b_U_Y * U + epsilon_Y
        logit_Y0 = intercept_Y + b_X_Y * 0 + b_U_Y * U + epsilon_Y
        p_Y1 = squasher_Y(logit_Y1)
        p_Y0 = squasher_Y(logit_Y0)
        ATE_true = np.mean(p_Y1 - p_Y0)
        PNS_true = np.mean(p_Y1 * (1 - p_Y0))

        return {
            'seed': seed,
            'intercept_X': intercept_X,
            'intercept_Y': intercept_Y,
            'b_U_X': b_U_X,
            'b_X_Y': b_X_Y,
            'b_U_Y': b_U_Y,
            'ATE_true': ATE_true,
            'PNS_true': PNS_true,
            'p_Y1_mean': p_Y1.mean(),
            'p_Y0_mean': p_Y0.mean(),
            'p_U': p_U,
            'U': U,
            'X': X,
            'Y': Y,
            'epsilon_X': epsilon_X,
            'sigma_X_vec': sigma_X_vec,
            'epsilon_Y': epsilon_Y,
            'sigma_Y_vec': sigma_Y_vec,
            'entropy_U': datagen_util.entropy_of_array(U),
            'entropy_X': datagen_util.entropy_of_array(X),
            'entropy_Y': datagen_util.entropy_of_array(Y),
            'squasher_X_name': squasher_X_name,
            'squasher_Y_name': squasher_Y_name,
            'heteroskedasticity_structure': 'sigma_i ~ |N(0,1)| for each unit'
        }