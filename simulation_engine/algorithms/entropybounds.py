import numpy as np
import math
import numpy as np
import cvxpy as cp

class EntropyBounds:
    """
    Class to calculate the entropy bounds for a given set of probabilities.
    Code taken from: 
    https://github.com/ziwei-jiang/Approximate-Causal-Effect-Identification-under-Weak-Confounding

    """
    @staticmethod
    def run_experiment_binaryIV_ATE(df, entr, method="cf"):
        """
        Run the EntropyBounds experiment.
        Parameters:
            df (pd.DataFrame): DataFrame containing the data for the experiment.
            entr (float): Upper bound on confounder entropy.
            method (str): Method to use for optimization ('cf' or 'cp').
        Returns:
            tuple: (lower_bound, upper_bound) from EntropyBounds
        """
        # Extract treatment and outcome variables
        X = df['X'].values
        Y = df['Y'].values

        # Compute ATE bounds
        return EntropyBounds._compute_ate_bounds(X, Y, entr, method)


    @staticmethod
    def _compute_ate_bounds(X, Y, entr, method="cf"):
        """
        Compute ATE bounds under entropy constraint.

        Parameters:
        - X: array-like of binary treatment assignments
        - Y: array-like of binary outcomes
        - entr: float, upper bound on confounder entropy
        - method: str, either 'cf' (counterfactual) or 'cp' (canonical partition)

        Returns:
        - (ate_lower, ate_upper): tuple of floats
        """
        # Select which optimizer to use
        if method == "cf":
            optimizer = EntropyBounds._optimization_cf
        elif method == "cp":
            optimizer = EntropyBounds._optimization_cp
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'cf' or 'cp'.")

        # Estimate P(X) and P(Y|X) from data
        x      = np.mean(X)
        x_bar  = 1 - x
        y_x    = np.mean(Y[X == 1])
        y_barx = 1 - y_x

        # Build observational joint P(Y,X)
        pyx = np.array([[y_x * x, y_barx * x],
                        [0.5 * x_bar, 0.5 * x_bar]]).T

        # Solve for bounds on P(Y=1 | do(X=1)) and P(Y=1 | do(X=0))
        lb11, ub11, _ = optimizer(pyx, ub=entr, p=1, q=1)
        lb10, ub10, _ = optimizer(pyx, ub=entr, p=1, q=0)

        # Calculate ATE bounds
        ate_lower = lb11 - ub10
        ate_upper = ub11 - lb10

        return ate_lower, ate_upper


    @staticmethod
    def _optimization_cf(pyx, ub=1, p =0, q=0):
        ##### for P(Y=p|do(X=q)) #####
        nx = pyx.shape[1]
        ny = pyx.shape[0]
        px = pyx.sum(axis=0)
        # uy_x: 1x(nx*ny) vector
        uy_x = cp.Variable(nx*ny)
        v1 = np.zeros(nx*ny)
        v1[p*nx:p*nx+nx] = px
        pydox = uy_x @ v1
        v2 = np.zeros((ny, nx*ny))
        for i in range(ny):
            v2[i, i*nx:(i+1)*nx] = px
        # qy: 1xny vector 
        qy = uy_x @ v2.T
        # qyx: 1x(nx*ny) vector
        qyx = qy @ v2
        v3 = np.zeros((nx*ny))
        for i in range(ny):
            v3[i*nx:(i+1)*nx] = px
        uyx = cp.multiply(uy_x, v3)
        dkl = cp.kl_div(uyx, qyx)/math.log(2)
        I = cp.sum(dkl)
        v4 = np.zeros((nx*ny, ny))
        for i in range(ny):
            v4[nx*i+q, i] = px[q]
        v5 = np.zeros((nx*ny, nx))
        for i in range(nx):
            v5[i:nx*ny:nx, i] = 1
        constraints  = [uy_x @ v4 == pyx[:,q],
                        uy_x@v5 == np.ones(nx), 
                        uy_x >= 0, uy_x <= 1,
                        I <= ub]
        max_obj = cp.Maximize(pydox)
        min_obj = cp.Minimize(pydox)
        t = 0
        max_prob = cp.Problem(max_obj, constraints)
        max_prob.solve(solver=cp.SCS)
        t += max_prob.solver_stats.solve_time
        min_prob = cp.Problem(min_obj, constraints)
        min_prob.solve(solver=cp.SCS)
        t += min_prob.solver_stats.solve_time
        return min_prob.value, max_prob.value, t


    @staticmethod
    def _optimization_cp(pyx, ub=1, p=0, q=0):
        ##### for P(Y=p|do(X=q)) #####
        nrx = pyx.shape[1]
        y_shape = pyx.shape[0]
        px = pyx.sum(axis=0)
        ry_shape = y_shape**nrx    
        ry_x = cp.Variable((ry_shape,nrx))
        v0 = np.zeros(ry_shape)
        ## assign the first half of the vector to be 1
        for i in range(0, ry_shape//2):
            v0[i] = 1
        v1 = 1 - v0
        pydox = ry_x@px @v0
        vx = np.zeros_like(px)
        vx[q] = px[q]
        ## make a vector of px with the same shape as ry_x with stack
        v4 = np.repeat(px[np.newaxis,:], ry_x.shape[0], axis=0)
        ryx = cp.multiply(ry_x, v4)
        ry = ry_x @ px
        qy = cp.vstack([ry]*nrx).T
        qyx = cp.multiply(qy, v4)
        dkl = cp.kl_div(ryx, qyx)/math.log(2)
        I = cp.sum(dkl)
        t = 0
        constraints = [cp.sum(ry_x@px)== 1, ry_x >= 0, ry_x <= 1, (ry_x@vx)@v0 == pyx[p,q], (ry_x@vx)@v1 == pyx[1:,0].sum(), I <= ub]
        max_prob = cp.Problem(cp.Maximize(pydox),constraints)
        max_prob.solve(solver=cp.SCS, verbose=False)
        ## get solverstats
        t += max_prob.solver_stats.solve_time
        min_prob = cp.Problem(cp.Minimize(pydox),constraints)
        min_prob.solve(solver=cp.SCS)
        t+= min_prob.solver_stats.solve_time
        return min_prob.value, max_prob.value, t