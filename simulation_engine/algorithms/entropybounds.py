import numpy as np
import math
import numpy as np
import cvxpy as cp
import pandas as pd

from simulation_engine.util.alg_util import AlgUtil

class EntropyBounds:
    """
    Class to calculate the entropy bounds for a given set of probabilities.
    Code taken from: 
    https://github.com/ziwei-jiang/Approximate-Causal-Effect-Identification-under-Weak-Confounding
    The PNS bounds are my own extension.

    """
    
    def bound(data, theta=0.5, query='ATE', method="cf",  randomize_theta=False, true_theta=False):
        """
        Compute entropy bounds for the ATE using the given method and entropy constraint.

        Args:
            method (str): Method to use for computing bounds. Options are 'cf' or 'cp'. Default is 'cf'.
            entr (float): Upper bound on confounder entropy. Default is 0.5.
            query (str): The query type (e.g., 'ATE' or 'PNS') to compute bounds for.

        Returns:
            Void: This method modifies the data DataFrame in place.
        """
        assert method in {'cf', 'cp'}, "Method must be either 'cf' or 'cp'"
        assert query in {'ATE', 'PNS'}, "Query must be either 'ATE' or 'PNS'"
   
        for idx, sim in data.iterrows():
            if randomize_theta:
                # Randomize theta
                theta = np.random.uniform(0, 1)
            elif true_theta:
                # Use the true theta from the simulation
                theta = sim['entropy_U']

            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X']})
            failed = False
            
            try:
                bound_lower, bound_upper = EntropyBounds.run_experiment_binaryIV(df, theta, query, method)
                
            except Exception as e:
                failed = True
            
            #Flatten bounds to trivial ceils
            if failed:
                bound_upper = AlgUtil.get_trivial_Ceils(query)[1] 
            if failed: 
                bound_lower = AlgUtil.get_trivial_Ceils(query)[0]

            bound_lower, bound_upper = AlgUtil.flatten_bounds_to_trivial_ceils(query, bound_lower, bound_upper, failed)


            bounds_valid = bound_lower <= sim[query+'_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            theta_rounded = f"{theta:.2f}"  # Always show two decimal places, e.g., 0.50
            if true_theta:
                theta_rounded = "trueTheta"
            elif randomize_theta:
                theta_rounded = "randomTheta"
            data.at[idx, f"{query}_entropybounds-{theta_rounded}_bound_lower"] = bound_lower
            data.at[idx, f"{query}_entropybounds-{theta_rounded}_bound_upper"] = bound_upper
            data.at[idx, f"{query}_entropybounds-{theta_rounded}_bound_valid"] = bounds_valid
            data.at[idx, f"{query}_entropybounds-{theta_rounded}_bound_width"] = bounds_width
            data.at[idx, f"{query}_entropybounds-{theta_rounded}_bound_failed"] = failed
            data.at[idx, f"{query}_entropybounds-{theta_rounded}_theta"] = theta


    @staticmethod
    def run_experiment_binaryIV(df, entr, query, method="cf"):
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

        if query == "ATE":
            # Compute ATE bounds
            return EntropyBounds._compute_ate_bounds(X, Y, entr, method)
        elif query == "PNS":
            # Compute PNS bounds
            return EntropyBounds._compute_pns_bounds(X, Y, entr)
        else:
            raise ValueError(f"Unknown query '{query}'. Choose 'ATE' or 'PNS'.")

    @staticmethod
    def _compute_pns_bounds(X, Y, entr):
        """
        Bounds on PNS = P(Y1=1, Y0=0) under an entropy-(mutual-information) cap.

        Parameters
        ----------
        X : 1-d array of 0/1 treatment assignments
        Y : 1-d array of 0/1 outcomes (same length as X)
        entr : float
            Upper bound on I((Y1,Y0);X)  (≤ entropy of latent confounder).

        Returns
        -------
        (lb, ub) : tuple of floats, the sharp lower and upper bounds on PNS.
        """
        # ---------- 1.  empirical quantities (constants) ----------
        p_x1 = float(np.mean(X))
        p_x0 = 1.0 - p_x1

        #  P(Y=1 | X)
        p_y1_x1 = float(np.mean(Y[X == 1])) if np.any(X == 1) else 0.0
        p_y1_x0 = float(np.mean(Y[X == 0])) if np.any(X == 0) else 0.0

        # ---------- 2.  decision variable : joint P(Y1,Y0,X) ----------
        # state order (row-major, X fastest):
        # (y1,y0,x) = (0,0,0) … (0,0,1) … (1,1,1)  ⇒  8 states
        q = cp.Variable(8)

        # ---------- 3.  basic probability constraints ----------
        constraints = [
            cp.sum(q) == 1,
            q >= 0
        ]

        # ---------- 4.  match the observable distribution P(Y=1 ,  X) ----------
        #   Y=1 with X=1  ⇔  Y1=1 ,  x=1   → indices 5 and 7
        constraints.append(q[5] + q[7] == p_y1_x1 * p_x1)

        #   Y=1 with X=0  ⇔  Y0=1 ,  x=0   → indices 2 and 6
        constraints.append(q[2] + q[6] == p_y1_x0 * p_x0)

        # ---------- 5.  mutual-information (entropy) constraint ----------
        #
        # selector S :  P(Y1,Y0) = S @ q    (4×8  constant)
        S = np.zeros((4, 8))
        S[0, [0, 1]] = 1          # (0,0)
        S[1, [2, 3]] = 1          # (0,1)
        S[2, [4, 5]] = 1          # (1,0)
        S[3, [6, 7]] = 1          # (1,1)
        p_y1y0 = S @ q            # length-4 affine expression

        # replicator T :  product dist  r  =  T @ p_y1y0   (8×4  constant)
        T = np.zeros((8, 4))
        for k in range(4):
            T[2 * k,     k] = p_x0      # x = 0
            T[2 * k + 1, k] = p_x1      # x = 1
        r = T @ p_y1y0                  # length-8 affine expression

        # KL-divergence  (I((Y1,Y0);X) in bits)
        kl_bits = cp.sum(cp.kl_div(q, r)) / math.log(2)
        constraints.append(kl_bits <= entr)

        # ---------- 6.  objective :  PNS = P(Y1=1 , Y0=0) ----------
        # indices with (y1=1, y0=0):  x=0 → 4 ,  x=1 → 5
        pns = q[4] + q[5]

        # ---------- 7.  optimise ----------
        lb_prob = cp.Problem(cp.Minimize(pns), constraints)
        ub_prob = cp.Problem(cp.Maximize(pns), constraints)
        lb_prob.solve(solver=cp.SCS)
        ub_prob.solve(solver=cp.SCS)

        return lb_prob.value, ub_prob.value



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
    

    @staticmethod
    def entropy_of_array(arr):
        """
        Calculate the empirical Shannon entropy of a numpy array in bits.
        """
        # Count occurrences of each unique value
        counts = np.bincount(arr)
        probabilities = counts / len(arr)
        
        # Filter out zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        
        # Calculate entropy in bits
        return -np.sum(probabilities * np.log2(probabilities))
