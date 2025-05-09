import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PlottingUtil:
    """
    A utility class for plotting and visualization.
    """

    @staticmethod
    def plot_smoothed_ate_vs_bounds(dataframe, algorithm='autobound', window=30):
        """
        Plot smoothed ATE_true and algorithm confidence intervals from the given dataframe.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing columns 'ATE_true', '<algorithm>_bound_lower', '<algorithm>_bound_upper', and 'b_X_Y'.
        algorithm (str): The name of the algorithm to use for bounds (e.g., 'autobound').
        window (int): The size of the rolling window for smoothing. Default is 30.

        Returns:
        None
        """
        # Create a copy of the dataframe to avoid modifying the original
        df = dataframe.copy()

        # Check if b_X_Y has varying values
        if dataframe['b_X_Y'].nunique() <= 1:
            print("Error: The 'b_X_Y' column has constant or invalid values. Cannot plot.")
            return

        # Smoothen the data using a rolling average
        df['ATE_true_smooth'] = df['ATE_true'].rolling(window=window, center=True).mean()
        df[f'{algorithm}_bound_lower_smooth'] = df[f'{algorithm}_bound_lower'].rolling(window=window, center=True).mean()
        df[f'{algorithm}_bound_upper_smooth'] = df[f'{algorithm}_bound_upper'].rolling(window=window, center=True).mean()

        # Plot the smoothed data
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='b_X_Y', y='ATE_true_smooth', label='$ATE_{true}$', color='blue')
        sns.lineplot(data=df, x='b_X_Y', y=f'{algorithm}_bound_lower_smooth', label='Lower Bound', color='orange')
        sns.lineplot(data=df, x='b_X_Y', y=f'{algorithm}_bound_upper_smooth', label='Upper Bound', color='green')
        plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.title(algorithm+' vs $ATE_{true}$ (smoothed out)')
        plt.xlabel('b_X_Y Coefficient')
        plt.ylabel('ATE Value')
        plt.legend()
        plt.grid(True)
        plt.show()