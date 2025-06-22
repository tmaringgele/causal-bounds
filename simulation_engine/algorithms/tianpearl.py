


import numpy as np
from simulation_engine.util.alg_util import AlgUtil


class TianPearl:
    def bound(data, query):
        """
        Compute Tian & Pearl bounds for a given query using only observed treatment (X) and outcome (Y).

        Supported queries:
            - 'ATE' : Average Treatment Effect
            - 'PNS' : Probability of Necessity and Sufficiency (without assuming exogeneity)

        Args:
            query (str): One of 'ATE' or 'PNS'

        Returns:
            Void: Modifies data DataFrame in place by adding bound columns.
        """
        assert query in {'ATE', 'PNS'}, "Query must be either 'ATE' or 'PNS'"

        for idx, sim in data.iterrows():
            X = np.array(sim['X'])
            Y = np.array(sim['Y'])
            failed = False

            try:
                if query == 'ATE':
                    p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.0
                    p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.0

                    # Bounds on P(Y=1 | do(X=1)) and do(X=0)
                    lower_do1 = p1
                    upper_do1 = 1 - p0
                    lower_do0 = p0
                    upper_do0 = 1 - p1

                    # ATE bounds are differences of those intervals
                    lower = lower_do1 - upper_do0
                    upper = upper_do1 - lower_do0

                elif query == 'PNS':
                    # Nonparametric bounds without assuming exogeneity
                    p_xy = np.mean((X == 1) & (Y == 1))
                    p_x0y0 = np.mean((X == 0) & (Y == 0))

                    lower = 0.0
                    upper = p_xy + p_x0y0

                # Ensure logical ordering
                lower, upper = min(lower, upper), max(lower, upper)

            except Exception:
                failed = True
                lower, upper = 0.0, 1.0  # fallback in case of failure

            # Flatten bounds to trivial ceils
            lower, upper = AlgUtil.flatten_bounds_to_trivial_ceils(query, lower, upper, failed)

            # Validity check
            if query == 'ATE':
                bounds_valid = lower <= sim['ATE_true'] <= upper
            else:
                bounds_valid = lower <= sim['PNS_true'] <= upper

            bounds_width = upper - lower

            data.at[idx, f'{query}_tianpearl_bound_lower'] = lower
            data.at[idx, f'{query}_tianpearl_bound_upper'] = upper
            data.at[idx, f'{query}_tianpearl_bound_width'] = bounds_width
            data.at[idx, f'{query}_tianpearl_bound_failed'] = failed
            data.at[idx, f'{query}_tianpearl_bound_valid'] = bounds_valid

        return data
