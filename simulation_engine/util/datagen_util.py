
import numpy as np
from scipy.stats import norm


class datagen_util:
    """
    Utility class for generating data for simulation scenarios.
    """


    @staticmethod
    def get_squashers():
        """
        Returns a dictionary of squashing functions that map real-valued logits to probabilities in [0, 1].

        Returns:
            dict: A dictionary with function names as keys and callable functions as values.
        """
        squashers = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),  # standard logistic
            "tanh_scaled": lambda x: 0.5 * (1 + np.tanh(x)),  # scaled tanh to [0, 1]
            "softplus": lambda x: np.log1p(np.exp(x)) / (1 + np.log1p(np.exp(x))),  # smooth capped softplus
            "probit": lambda x: norm.cdf(x),  # Gaussian CDF
        }
        return squashers
    
    @staticmethod
    def _sample_p_with_uniform_entropy():
        """
        Sample a Bernoulli parameter p such that the entropy H(Ber(p)) is uniformly distributed in [0,1].

        Returns
        -------
        float : p ∈ [0, 1] with H(Ber(p)) ~ Uniform(0,1)
        """
        import scipy.optimize

        def binary_entropy(p):
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        def inverse_entropy(h_target):
            # Find p in [0, 0.5] such that H(p) = h_target
            f = lambda p: binary_entropy(p) - h_target
            return scipy.optimize.bisect(f, 1e-6, 0.5 - 1e-6)

        # Step 1: Sample target entropy
        h = np.random.uniform(0, 1)
        # Step 2: Find p in [0, 0.5]
        p = inverse_entropy(h)
        # Step 3: Reflect with 50% probability
        return p if np.random.rand() < 0.5 else 1 - p


    @staticmethod
    def safe_entropy(arr):
        # Convert float arrays to int64 for entropy calculation
        arr = np.asarray(arr).astype(np.int64)
        return datagen_util.entropy_of_array(arr)

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
    
    @staticmethod
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
