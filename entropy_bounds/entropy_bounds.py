import numpy as np
from entropy_bounds.utils import optimization_cf, optimization_cp

def compute_ate_bounds(X, Y, entr, method="cf"):
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
        optimizer = optimization_cf
    elif method == "cp":
        optimizer = optimization_cp
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
