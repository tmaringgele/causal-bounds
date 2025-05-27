import numpy as np
import pandas as pd
import pulp

class ZahngBareinboim:
    @staticmethod
    def clean_data(data):
        def unwrap(val):
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return val.item()
                else:
                    raise ValueError(f"Array entry has more than one element: {val}")
            return val

        for col in ['Z', 'X', 'Y']:
            data[col] = data[col].apply(unwrap)
        return data

    @staticmethod
    def expand_row_to_long(row):
        arrays = {
            'X': row['X'],
            'Z': row['Z'],
            'Y': row['Y']
        }
        n = len(arrays['X'])
        return pd.DataFrame({k: arrays[k] for k in arrays}, index=range(n))

    @staticmethod
    def compute_observational_distributions(data):
        Z_values = sorted(data['Z'].unique().tolist())
        X_values = sorted(data['X'].unique().tolist())
        P_x_given_z = {}
        E_Y_given_xz = {}

        for z in Z_values:
            data_z = data[data['Z'] == z]
            n_z = len(data_z)
            for x in X_values:
                data_xz = data_z[data_z['X'] == x]
                n_xz = len(data_xz)
                p_x_given_z = n_xz / n_z if n_z > 0 else 0
                e_y_given_xz = data_xz['Y'].mean() if n_xz > 0 else 0
                P_x_given_z[(x, z)] = p_x_given_z
                E_Y_given_xz[(x, z)] = e_y_given_xz

        return Z_values, X_values, P_x_given_z, E_Y_given_xz

    @staticmethod
    def generate_compliance_types(X_values, Z_values):
        from itertools import product
        return [tuple(x) for x in product(X_values, repeat=len(Z_values))]

    @staticmethod
    def solve_bounds(dataset_row, target_x):
        import pulp

        if isinstance(dataset_row, pd.Series):
            data = ZahngBareinboim.expand_row_to_long(dataset_row)
        else:
            raise TypeError("Input must be a row (Series) where columns 'Z', 'X', 'Y' are numpy arrays.")

        data = ZahngBareinboim.clean_data(data)
        Z_values, X_values, P_x_given_z, E_Y_given_xz = ZahngBareinboim.compute_observational_distributions(data)
        compliance_types = ZahngBareinboim.generate_compliance_types(X_values, Z_values)

        lp_min = pulp.LpProblem("Minimize_EYx", pulp.LpMinimize)
        lp_max = pulp.LpProblem("Maximize_EYx", pulp.LpMaximize)

        P = {c: pulp.LpVariable(f"P_{c}", lowBound=0, upBound=1) for c in compliance_types}
        EY = {c: pulp.LpVariable(f"EY_{c}", lowBound=0, upBound=1) for c in compliance_types}
        A = {c: pulp.LpVariable(f"A_{c}", lowBound=0, upBound=1) for c in compliance_types}  # auxiliary for EY[c] * P[c]

        # Define A[c] ≈ EY[c] * P[c] using McCormick envelopes
        for c in compliance_types:
            lp_min += A[c] <= P[c]
            lp_min += A[c] <= EY[c]
            lp_min += A[c] >= EY[c] + P[c] - 1
            lp_min += A[c] >= 0

            lp_max += A[c] <= P[c]
            lp_max += A[c] <= EY[c]
            lp_max += A[c] >= EY[c] + P[c] - 1
            lp_max += A[c] >= 0

        # Normalization
        lp_min += pulp.lpSum(P[c] for c in compliance_types) == 1
        lp_max += pulp.lpSum(P[c] for c in compliance_types) == 1

        for z_index, z in enumerate(Z_values):
            for x in X_values:
                indicator = [int(c[z_index] == x) for c in compliance_types]

                lp_min += pulp.lpSum(indicator[i] * P[c] for i, c in enumerate(compliance_types)) == P_x_given_z[(x, z)]
                lp_max += pulp.lpSum(indicator[i] * P[c] for i, c in enumerate(compliance_types)) == P_x_given_z[(x, z)]

                lhs_min = pulp.lpSum(indicator[i] * A[c] for i, c in enumerate(compliance_types))
                lhs_max = pulp.lpSum(indicator[i] * A[c] for i, c in enumerate(compliance_types))
                rhs = E_Y_given_xz[(x, z)] * P_x_given_z[(x, z)]

                lp_min += lhs_min == rhs
                lp_max += lhs_max == rhs

        # Objective
        lp_min += pulp.lpSum((int(c[target_x] == x) * A[c]) for c in compliance_types)
        lp_max += pulp.lpSum((int(c[target_x] == x) * A[c]) for c in compliance_types)

        lp_min.solve()
        lp_max.solve()

        return {
            "lower_bound": pulp.value(lp_min.objective),
            "upper_bound": pulp.value(lp_max.objective)
        }
