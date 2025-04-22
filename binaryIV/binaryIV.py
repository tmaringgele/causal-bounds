# Required imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pick_from_bimodal(n=1, mu1=1, sigma1=0.5, mu2=-1, sigma2=0.5):
    """
    Generate samples from a bimodal distribution.

    Args:
        n (int): Number of samples to pick. Default is 1.
        mu1 (float): Mean of the first normal distribution. Default is 1.
        sigma1 (float): Standard deviation of the first normal distribution. Default is 0.5.
        mu2 (float): Mean of the second normal distribution. Default is -1.
        sigma2 (float): Standard deviation of the second normal distribution. Default is 0.5.

    Returns:
        float or np.ndarray: A single sample if n=1, otherwise an array of n samples.
    """
    N=20000
    X1 = np.random.normal(mu1, sigma1, N)
    X2 = np.random.normal(mu2, sigma2, N)
    X = np.concatenate([X1, X2])
    # sns.histplot(X, bins=30, kde=True)
    # Pick n random samples from the bimodal distribution
    samples = np.random.choice(X, n, replace=False)
    # if n is one sample, return the sample as a single value
    if n == 1:
        return samples[0]
    return samples

    # # Example usage
    # samples = pick_from_bimodal(n=20000, mu1=1, sigma1=0.5, mu2=-1, sigma2=0.5)
    # print(samples)
    # sns.histplot(samples, bins=30, kde=True)
    # plt.show() 

def simulate_deterministic_data_with_probabilistic_ate(
    n=500,
    seed=None,
    b_U_X=np.random.normal(0, 1, 1),
    b_U_Y=np.random.normal(0, 1, 1),
    b_Z=pick_from_bimodal(),
    b_X_Y=pick_from_bimodal(),
    intercept_X=0,
    intercept_Y=0
):
    """
    Simulate deterministic (binary) data for causal analysis, 
    while computing the Average Treatment Effect (ATE) from smooth logistic potential outcome probabilities.

    Args:
        n (int): Number of samples to generate. Default is 500.
        seed (int, optional): Random seed for reproducibility. Default is None.
        b_U_X (float): Coefficient for the effect of unobserved confounder U on X. Default is drawn from N(0, 1).
        b_U_Y (float): Coefficient for the effect of unobserved confounder U on Y. Default is drawn from N(0, 1).
        b_Z (float): Coefficient for the effect of instrument Z on X. Default is drawn from a bimodal distribution.
        b_X_Y (float): Coefficient for the effect of treatment X on Y. Default is drawn from a bimodal distribution.
        intercept_X (float): Intercept for the logistic model of X. Default is 0.
        intercept_Y (float): Intercept for the logistic model of Y. Default is 0.

    Returns:
        dict: A dictionary containing:
            - seed (int): The random seed used.
            - intercept_X (float): Intercept for X.
            - intercept_Y (float): Intercept for Y.
            - b_Z (float): Coefficient for Z.
            - b_U_X (float): Coefficient for U on X.
            - b_X_Y (float): Coefficient for X on Y.
            - b_U_Y (float): Coefficient for U on Y.
            - ATE_true (float): True Average Treatment Effect.
            - p_Y1 (np.ndarray): Probabilities of Y=1 under treatment.
            - p_Y0 (np.ndarray): Probabilities of Y=1 under control.
            - Z (np.ndarray): Instrument variable.
            - U (np.ndarray): Unobserved confounder.
            - X (np.ndarray): Treatment assignment.
            - Y (np.ndarray): Outcome variable.
    """
    if seed is None:
        seed = np.random.randint(0, 1e6)
    np.random.seed(seed)
    b_U_X=np.random.normal(0, 1, 1),
    b_U_Y=np.random.normal(0, 1, 1),
    b_Z=pick_from_bimodal(),
    b_X_Y=pick_from_bimodal(),
    intercept_X=0,
    intercept_Y=0
    print(f"Seed: {seed}, b_U_X: {b_U_X}, b_U_Y: {b_U_Y}, b_Z: {b_Z}, b_X_Y: {b_X_Y}")




    # Binary variables
    Z = np.random.binomial(1, 0.5, size=n)
    U = np.random.binomial(1, 0.5, size=n)

    # Treatment assignment
    logit_X = intercept_X + b_Z * Z + b_U_X * U
    p_X = 1 / (1 + np.exp(-logit_X))
    X = np.random.binomial(1, p_X)

    # print("b_Z:", b_Z)
    # print("b_U_X:", b_U_X)
    # print("Mean of p_X:", np.mean(p_X))
    # print("Mean of X:", np.mean(X))

    # Deterministic outcome
    logit_Y = intercept_Y + b_X_Y * X + b_U_Y * U
    p_Y = 1 / (1 + np.exp(-logit_Y))
    Y = np.random.binomial(1, p_Y)

    # Probabilistic potential outcomes
    logit_Y1 = intercept_Y + b_X_Y * 1 + b_U_Y * U
    logit_Y0 = intercept_Y + b_X_Y * 0 + b_U_Y * U
    p_Y1 = 1 / (1 + np.exp(-logit_Y1))
    p_Y0 = 1 / (1 + np.exp(-logit_Y0))
    ATE_true = np.mean(p_Y1 - p_Y0)

    return {
        'seed': seed,
        'intercept_X': intercept_X,
        'intercept_Y': intercept_Y,
        'b_Z': b_Z,
        'b_U_X': b_U_X,
        'b_X_Y': b_X_Y,
        'b_U_Y': b_U_Y,
        'ATE_true': ATE_true,
        'p_Y1': p_Y1,
        'p_Y0': p_Y0,
        'Z': Z,
        'U': U,
        'X': X,
        'Y': Y
    }
