import numpy as np
import pandas as pd
import pulp
from itertools import product
from simulation_engine.util.alg_util import AlgUtil

class ZhangBareinboim:

    ALG_NAME = "zhangbareinboim"

    @staticmethod
    def bound_ATE(data):
        """
        Compute Zhang-Bareinboim bounds on the Average Treatment Effect (ATE)

        Args:
            data (pd.Series or pd.DataFrame): Dataframe of Simulations with columns 'Z', 'X', 'Y'.
        """
        query = "ATE"
        for idx, sim in data.iterrows():

            failed = False
            try:
                # Compute P(Y | do(x=1))
                b1  = ZhangBareinboim._bound_causal_effect(sim, target_x=1)
                # Compute P(Y | do(x=0))
                b0  = ZhangBareinboim._bound_causal_effect(sim, target_x=0)
                bound_lower = b1["lower_bound"] - b0["upper_bound"]
                bound_upper = b1["upper_bound"] - b0["lower_bound"]
            except Exception as e:
                print(f"Error in ZhangBareinboim: {e}")
                failed = True
            #Flatten bounds to trivial ceils
            if failed:
                bound_upper = AlgUtil.get_trivial_Ceils(query)[1] 
                bound_lower = AlgUtil.get_trivial_Ceils(query)[0]
            
            bound_lower, bound_upper = AlgUtil.flatten_bounds_to_trivial_ceils(query, bound_lower, bound_upper, failed)



            bounds_valid = bound_lower <= sim[query+'_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            
            data.at[idx, query+'_'+ZhangBareinboim.ALG_NAME+'_bound_lower'] = bound_lower
            data.at[idx, query+'_'+ZhangBareinboim.ALG_NAME+'_bound_upper'] = bound_upper
            data.at[idx, query+'_'+ZhangBareinboim.ALG_NAME+'_bound_valid'] = bounds_valid
            data.at[idx, query+'_'+ZhangBareinboim.ALG_NAME+'_bound_width'] = bounds_width
            data.at[idx, query+'_'+ZhangBareinboim.ALG_NAME+'_bound_failed'] = failed

        return data


    @staticmethod
    def _prepare_data(data):
        """
        Convert raw observational data into a pandas DataFrame with columns Z, X, Y.
        Accepts either a pandas Series (with entries 'Z', 'X', 'Y' as numpy arrays)
        or an already-expanded DataFrame.
        """
        if isinstance(data, pd.Series):
            # Expand the series of arrays into a DataFrame (one row per observation)
            df = pd.DataFrame({
                "Z": data["Z"],
                "X": data["X"],
                "Y": data["Y"]
            })
        else:
            # Assume it's already a DataFrame with columns Z, X, Y
            df = data.copy()
        # Ensure any 0-dim numpy arrays are converted to scalars (just in case)
        for col in ["Z", "X", "Y"]:
            df[col] = df[col].apply(
                lambda v: v.item() if isinstance(v, np.ndarray) and v.shape == () else v
            )
        return df

    @staticmethod
    def _compute_observational_stats(df):
        """
        Compute the empirical P(X=x | Z=z) and E[Y | X=x, Z=z] from the data frame.
        Returns:
        Z_vals: sorted list of unique Z values
        X_vals: sorted list of unique X values
        P_x_given_z: dict mapping (x,z) -> probability P(X=x | Z=z)
        EY_given_xz: dict mapping (x,z) -> conditional mean E[Y | X=x, Z=z]
        """
        Z_vals = sorted(df["Z"].unique().tolist())
        X_vals = sorted(df["X"].unique().tolist())
        P_x_given_z = {}
        EY_given_xz = {}
        for z in Z_vals:
            df_z = df[df["Z"] == z]
            n_z = len(df_z)
            for x in X_vals:
                df_xz = df_z[df_z["X"] == x]
                n_xz = len(df_xz)
                # P(X=x | Z=z) = count(X=x,Z=z) / count(Z=z)
                P_x_given_z[(x, z)] = (n_xz / n_z) if n_z > 0 else 0.0
                # E[Y | X=x, Z=z] = average Y in the subset where X=x, Z=z
                EY_given_xz[(x, z)] = df_xz["Y"].mean() if n_xz > 0 else 0.0
        return Z_vals, X_vals, P_x_given_z, EY_given_xz

    @staticmethod
    def _enumerate_compliance_types(X_vals, Z_vals):
        """
        Enumerate all possible compliance types as tuples.
        Each compliance type is a |Z_vals|-length tuple (t_z1, t_z2, ..., t_zm)
        representing the treatment X that would be taken for each value of Z.
        """
        return [tuple(pattern) for pattern in product(X_vals, repeat=len(Z_vals))]

    @staticmethod
    def _bound_causal_effect(simulation, target_x):
        """
        Solve the LP to find bounds on E[Y_target_x] (the expected outcome if X is set to target_x).
        Returns a dictionary with 'lower_bound' and 'upper_bound'.
        Throws an exception if the LPs are invalid or produce an error.
        """
        df = ZhangBareinboim._prepare_data(simulation)
        Z_vals, X_vals, P_x_given_z, EY_given_xz = ZhangBareinboim._compute_observational_stats(df)
        compliance_types = ZhangBareinboim._enumerate_compliance_types(X_vals, Z_vals)
        
        # Define LP problems for min and max
        lp_min = pulp.LpProblem("Min_E[Y{}_bound]".format(target_x), pulp.LpMinimize)
        lp_max = pulp.LpProblem("Max_E[Y{}_bound]".format(target_x), pulp.LpMaximize)
        
        # Define LP variables:
        # P[c] for each compliance type c, and Q[c,x] for each type and each treatment value
        P_vars = {c: pulp.LpVariable(f"P_{c}", lowBound=0, upBound=1) 
                for c in compliance_types}
        Q_vars = {(c, x): pulp.LpVariable(f"Q_{c}_{x}", lowBound=0, upBound=1) 
                for c in compliance_types for x in X_vals}
        
        # Constraint: Q(c,x) <= P(c) for all c, x  (ensures 0 <= E[Y_x|c] <= 1)
        for c in compliance_types:
            for x in X_vals:
                lp_min += Q_vars[(c, x)] <= P_vars[c]
                lp_max += Q_vars[(c, x)] <= P_vars[c]
        
        # Constraint: Sum_c P(c) = 1  (all compliance type probabilities sum to 1)
        lp_min += pulp.lpSum(P_vars[c] for c in compliance_types) == 1
        lp_max += pulp.lpSum(P_vars[c] for c in compliance_types) == 1
        
        # Constraints to match observed distribution and outcomes:
        for z in Z_vals:
            # index of z in the sorted Z_vals list (to pick the corresponding element of compliance type tuple)
            z_index = Z_vals.index(z)
            for x in X_vals:
                # All compliance types c where c(z_index) == x
                matching_types = [c for c in compliance_types if c[z_index] == x]
                # Match P(X=x | Z=z):
                lp_min += pulp.lpSum(P_vars[c] for c in matching_types) == P_x_given_z[(x, z)]
                lp_max += pulp.lpSum(P_vars[c] for c in matching_types) == P_x_given_z[(x, z)]
                # Match E[Y | X=x, Z=z] * P(X=x | Z=z):
                # The right-hand side is the observed joint proportion of (X=x, Z=z) times the average Y in that cell
                rhs = P_x_given_z[(x, z)] * EY_given_xz[(x, z)]
                lp_min += pulp.lpSum(Q_vars[(c, x)] for c in matching_types) == rhs
                lp_max += pulp.lpSum(Q_vars[(c, x)] for c in matching_types) == rhs
        
        # Objective: E[Y_target_x] = sum_c Q(c, target_x)
        lp_min += pulp.lpSum(Q_vars[(c, target_x)] for c in compliance_types)
        lp_max += pulp.lpSum(Q_vars[(c, target_x)] for c in compliance_types)
        


        # Solve the two LPs
        min_status = lp_min.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[lp_min.status] != "Optimal":
            raise Exception(f"LP for lower bound is invalid or infeasible: status={pulp.LpStatus[lp_min.status]}")
        lower = pulp.value(lp_min.objective)

        max_status = lp_max.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[lp_max.status] != "Optimal":
            raise Exception(f"LP for upper bound is invalid or infeasible: status={pulp.LpStatus[lp_max.status]}")
        upper = pulp.value(lp_max.objective)

        return {"lower_bound": lower, "upper_bound": upper}

