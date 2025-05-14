from rpy2.robjects import r, globalenv, IntVector, FloatVector


import rpy2.robjects as robjects
from rpy2.robjects.environments import Environment
from rpy2.robjects.packages import importr
from rpy2.robjects import IntVector, FloatVector
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
from io import StringIO
import pandas as pd

class Causaloptim:


    @staticmethod
    def bound_binaryIV(query, data):
        # Define Scenario and target quantity
        graph_str = "(Z -+ X, X -+ Y, Ur -+ X, Ur -+ Y)"
        leftside = [1, 0, 0, 0]
        latent = [0, 0, 0, 1]
        nvals = [2, 2, 2, 2]
        rlconnect = [0, 0, 0, 0]
        monotone = [0, 0, 0, 0]

        importr('causaloptim')
        importr('base')

        for idx, sim in data.iterrows():
            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            failed = False
            # try:

            result = Causaloptim._run_experiment(query, graph_str, leftside, latent, nvals, rlconnect, monotone, df)
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


            bounds_valid = bound_lower <= sim[query+'_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            
            data.at[idx, query+'_causaloptim_bound_lower'] = bound_lower
            data.at[idx, query+'_causaloptim_bound_upper'] = bound_upper
            data.at[idx, query+'_causaloptim_bound_valid'] = bounds_valid
            data.at[idx, query+'_causaloptim_bound_width'] = bounds_width
            data.at[idx, query+'_causaloptim_bound_failed'] = failed

        return data

    @staticmethod
    def _run_experiment(query, graph_str, leftside, latent, nvals, rlconnect, monotone, df):
        """
        Run a complete causal bounds experiment using R's global environment.
        
        Note: This function is not thread-safe. It resets the global R environment
        using rm(list=ls()) and closeAllConnections().

        Parameters:
            graph_str: String defining the DAG in igraph syntax
            leftside, latent, nvals: vertex attributes
            rlconnect, monotone: edge attributes
            prob_dict: dictionary of probabilities, e.g., {'p00_0': 0.2, ...}

        Returns:
            tuple: (lower_bound, upper_bound) from causaloptim
        """

        r_output = StringIO()

        # Set up a custom logger to capture R output
        handler = logging.StreamHandler(r_output)
        rpy2_logger.addHandler(handler)
        rpy2_logger.setLevel(logging.INFO)  # Adjust if necessary


        prob_dict = Causaloptim._extract_prob_dict(df)

        # Reset R state (clears variables, closes open files/connections)
        r('closeAllConnections(); rm(list=ls())')

        # Load required libraries (disabled, uncomment if needed)
        #importr("igraph")
        #importr("causaloptim")

        # Create the graph
        r(f'graph <- igraph::graph_from_literal{graph_str}')

        # Assign vectors to R
        globalenv["leftside"] = IntVector(leftside)
        globalenv["latent"] = IntVector(latent)
        globalenv["nvals"] = IntVector(nvals)
        globalenv["rlconnect"] = IntVector(rlconnect)
        globalenv["monotone"] = IntVector(monotone)

        r('V(graph)$leftside <- leftside')
        r('V(graph)$latent <- latent')
        r('V(graph)$nvals <- nvals')
        r('E(graph)$rlconnect <- rlconnect')
        r('E(graph)$edge.monotone <- monotone')

        # Inject probabilities
        for key, val in prob_dict.items():
            globalenv[key] = FloatVector([val])

        if query == "ATE":
            r("""
                query <- "p{Y(X = 1) = 1} - p{Y(X = 0) = 1}"
              """)
        elif query == "PNS":
            r("""
                query <- "p{Y(X = 1) = 1; Y(X = 0) = 0}"
              """)
        else:
            raise ValueError("Query must be either 'ATE' or 'PNS'")


        # Compute bounds
        r("""
            obj <- analyze_graph(graph, constraints = NULL, effectt = query)
            bounds <- optimize_effect_2(obj)
            boundsfunc <- interpret_bounds(bounds = bounds$bounds, obj$parameters)
            bounds_result <- boundsfunc(
            p00_0 = p00_0, p00_1 = p00_1,
            p10_0 = p10_0, p10_1 = p10_1,
            p01_0 = p01_0, p01_1 = p01_1,
            p11_0 = p11_0, p11_1 = p11_1
            )
        """)

        failed = False
        captured = r_output.getvalue()
        if 'Invalid' in captured:
            failed = True
            
        # Fetch and return result
        bounds = r("bounds_result")
        lb = bounds[0][0]
        ub = bounds[1][0]

        return {'lower_bound': lb, 'upper_bound': ub, 'failed': failed}

    @staticmethod
    def _extract_prob_dict(df):
        """
        Given a DataFrame with binary columns Y, X, Z,
        compute the conditional probabilities P(Y, X | Z) and return
        a dictionary in the format expected by causaloptim.

        Parameters:
            df (pd.DataFrame): must contain columns 'Y', 'X', and 'Z' with binary values

        Returns:
            dict: keys are of form 'pYX_Z', e.g. 'p00_0', 'p11_1'
        """
        assert set(['Y', 'X', 'Z']).issubset(df.columns), "df must have Y, X, Z columns"
        
        # Count joint occurrences
        joint_counts = df.groupby(['Y', 'X', 'Z']).size().reset_index(name='count')

        # Count total occurrences of each Z
        z_counts = df['Z'].value_counts().to_dict()

        # Compute conditional probabilities
        joint_counts['p_yx_z'] = joint_counts.apply(
            lambda row: row['count'] / z_counts[row['Z']], axis=1
        )

        # Convert to lookup dict
        marg_dict = {
            (int(row.Y), int(row.X), int(row.Z)): row.p_yx_z
            for _, row in joint_counts.iterrows()
        }

        # Helper to safely extract or return 0
        def get_prob(y, x, z):
            return marg_dict.get((y, x, z), 0.0)

        # Build final dictionary in causaloptim format
        return {
            'p00_0': get_prob(0, 0, 0),
            'p10_0': get_prob(1, 0, 0),
            'p01_0': get_prob(0, 1, 0),
            'p11_0': get_prob(1, 1, 0),
            'p00_1': get_prob(0, 0, 1),
            'p10_1': get_prob(1, 0, 1),
            'p01_1': get_prob(0, 1, 1),
            'p11_1': get_prob(1, 1, 1),
        }