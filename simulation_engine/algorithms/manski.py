

import numpy as np
from simulation_engine.util.alg_util import AlgUtil


class Manski:

    def bound_ATE(data):
        """
        Compute Manski-style bounds for a given query using only observed treatment (X) and outcome (Y).
        
        Supported queries:
            - 'ATE' : Average Treatment Effect
            - 'PNS' : Probability of Necessity and Sufficiency

        Args:
            query (str): One of 'ATE' or 'PNS'

        Returns:
            Void: Modifies data DataFrame in place by adding bound columns.
        """
        for idx, sim in data.iterrows():
            X = np.array(sim['X'])
            Y = np.array(sim['Y'])
            failed = False

            try:
                p1 = np.mean(Y[X == 1]) if np.any(X == 1) else 0.0
                p0 = np.mean(Y[X == 0]) if np.any(X == 0) else 0.0

                lower = p1 - p0 - 1
                upper = p1 - p0 + 1
                lower = max(lower, -1)
                upper = min(upper, 1)

                # Ensure logical ordering
                lower, upper = min(lower, upper), max(lower, upper)

            except Exception:
                failed = True

            # Flatten bounds to trivial ceils
            AlgUtil.flatten_bounds_to_trivial_ceils('ATE', lower, upper, failed)

            # Validity check only makes sense for ATE (if ATE_true is in the data)
            bounds_valid = lower <= sim['ATE_true'] <= upper


            bounds_width = upper - lower

            data.at[idx, f'ATE_manski_bound_lower'] = lower
            data.at[idx, f'ATE_manski_bound_upper'] = upper
            data.at[idx, f'ATE_manski_bound_width'] = bounds_width
            data.at[idx, f'ATE_manski_bound_failed'] = failed
            data.at[idx, f'ATE_manski_bound_valid'] = bounds_valid

        return data
        
