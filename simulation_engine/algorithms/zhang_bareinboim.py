import numpy as np
import pandas as pd
import pulp
from itertools import product

class ZhangBareinboim:

    @staticmethod
    def prepare_data(data):
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
    def compute_observational_stats(df):
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
    def enumerate_compliance_types(X_vals, Z_vals):
        """
        Enumerate all possible compliance types as tuples.
        Each compliance type is a |Z_vals|-length tuple (t_z1, t_z2, ..., t_zm)
        representing the treatment X that would be taken for each value of Z.
        """
        return [tuple(pattern) for pattern in product(X_vals, repeat=len(Z_vals))]

    @staticmethod
    def bound_causal_effect(data, target_x):
        """
        Solve the LP to find bounds on E[Y_target_x] (the expected outcome if X is set to target_x).
        Returns a dictionary with 'lower_bound' and 'upper_bound'.
        """
        df = ZhangBareinboim.prepare_data(data)
        Z_vals, X_vals, P_x_given_z, EY_given_xz = ZhangBareinboim.compute_observational_stats(df)
        compliance_types = ZhangBareinboim.enumerate_compliance_types(X_vals, Z_vals)
        
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
        lp_min.solve(pulp.PULP_CBC_CMD(msg=False))
        lower = pulp.value(lp_min.objective)      # ← keep it safe

        lp_max.solve(pulp.PULP_CBC_CMD(msg=False))
        upper = pulp.value(lp_max.objective)

        return {"lower_bound": lower, "upper_bound": upper}
                

class ZhangBareinboimOLD:

    # ---------- helpers from the previous class ----------
    @staticmethod
    def _expand_row_to_long(row):
        """unpack one row that stores Z,X,Y as numpy arrays into a long DF"""
        return pd.DataFrame({"Z": row["Z"], "X": row["X"], "Y": row["Y"]})

    @staticmethod
    def _clean_data(df):
        """turn 0-d arrays into scalars (just in case)"""
        for col in ["Z", "X", "Y"]:
            df[col] = df[col].apply(lambda v: v.item() if isinstance(v, np.ndarray) and v.size == 1 else v)
        return df

    @staticmethod
    def _obs_tables(df):
        Zvals = sorted(df["Z"].unique().tolist())
        Xvals = sorted(df["X"].unique().tolist())
        P_x_given_z, EY_given_xz = {}, {}
        for z in Zvals:
            block = df[df["Z"] == z]
            n_z   = len(block)
            for x in Xvals:
                blk2      = block[block["X"] == x]
                n_xz      = len(blk2)
                P_x_given_z[(x, z)]  = n_xz / n_z if n_z else 0.0
                EY_given_xz[(x, z)]  = blk2["Y"].mean() if n_xz else 0.0
        return Zvals, Xvals, P_x_given_z, EY_given_xz

    @staticmethod
    def _compliance_types(Xvals, Zvals):
        """all |Z|-tuples over X"""
        return [tuple(t) for t in product(Xvals, repeat=len(Zvals))]

    # ------------------------------------------------------

    @staticmethod
    def solve_bounds_exact(row, target_x=1):
        """
        Bounds on  E[Y_target_x]  using the exact LP of Zhang & Bareinboim (R-61).
        `row` must be a pandas.Series where row['Z'], row['X'], row['Y'] are nd-arrays
        """
        data = ZahngBareinboim._clean_data(ZahngBareinboim._expand_row_to_long(row))

        Zvals, Xvals, P_x_given_z, EY_given_xz = ZahngBareinboim._obs_tables(data)
        C_types = ZahngBareinboim._compliance_types(Xvals, Zvals)

        # --- two LPs: min & max -----------------------------------------------
        lp_min = pulp.LpProblem("MinEYx_min", pulp.LpMinimize)
        lp_max = pulp.LpProblem("MaxEYx_max", pulp.LpMaximize)

        # P[c] and Q[(c,x)] variables
        P = {c: pulp.LpVariable(f"P_{c}", lowBound=0, upBound=1) for c in C_types}
        Q = {(c, x):
            pulp.LpVariable(f"Q_{c}_{x}", lowBound=0, upBound=1)
            for c in C_types for x in Xvals}

        # Q[c,x] ≤ P[c]
        for c in C_types:
            for x in Xvals:
                lp_min += Q[(c, x)] <= P[c]
                lp_max += Q[(c, x)] <= P[c]

        # Σ P[c] = 1
        lp_min += pulp.lpSum(P[c] for c in C_types) == 1
        lp_max += pulp.lpSum(P[c] for c in C_types) == 1

        # observational constraints
        for z_idx, z in enumerate(Zvals):
            for x in Xvals:
                match = [c for c in C_types if c[z_idx] == x]

                # P(X=x | Z=z)
                lp_min += pulp.lpSum(P[c] for c in match) == P_x_given_z[(x, z)]
                lp_max += pulp.lpSum(P[c] for c in match) == P_x_given_z[(x, z)]

                # E[Y | x,z] · P(X=x | Z=z)
                rhs = EY_given_xz[(x, z)] * P_x_given_z[(x, z)]
                lp_min += pulp.lpSum(Q[(c, x)] for c in match) == rhs
                lp_max += pulp.lpSum(Q[(c, x)] for c in match) == rhs

        # --- objective:   E[Y_target_x] = Σ_c Q[(c, target_x)]  ---
        lp_min += pulp.lpSum(Q[(c, target_x)] for c in C_types)
        lp_max += pulp.lpSum(Q[(c, target_x)] for c in C_types)


        lp_min.solve(pulp.PULP_CBC_CMD(msg=False))
        lp_max.solve(pulp.PULP_CBC_CMD(msg=False))

        return {
            "lower_bound": pulp.value(lp_min.objective),
            "upper_bound": pulp.value(lp_max.objective)
        }

    # small convenience for ATE
    @staticmethod
    def solve_ate_bounds(row):
        b1 = ZahngBareinboim.solve_bounds_exact(row, target_x=1)
        b0 = ZahngBareinboim.solve_bounds_exact(row, target_x=0)
        return {
            "lower_bound": b1["lower_bound"] - b0["upper_bound"],
            "upper_bound": b1["upper_bound"] - b0["lower_bound"]
        }
