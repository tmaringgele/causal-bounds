
import numpy as np


class datagen_util:
    """
    Utility class for generating data for simulation scenarios.
    """
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
