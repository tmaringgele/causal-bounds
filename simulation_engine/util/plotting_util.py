import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PlottingUtil:
    """
    A utility class for plotting and visualization.
    """

    @staticmethod
    def print_bound_statistics(dataframe, algorithms=['autobound', 'causaloptim']):
        """
        Print statistics of the bounds for the given algorithms.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing bound information.
        algorithms (list): List of algorithm names to print statistics for.

        Returns:
        None
        """
        for algorithm in algorithms:
            if f'{algorithm}_bound_valid' in dataframe.columns:
                failed_bounds = dataframe[dataframe[f'{algorithm}_bound_failed']].shape[0]
                without_failed = dataframe[dataframe[f'{algorithm}_bound_failed'] == False]
                invalid_bounds = without_failed[without_failed[f'{algorithm}_bound_valid'] == False].shape[0]
                without__failed_and_invalid = without_failed[without_failed[f'{algorithm}_bound_valid'] == True]
                print(f"Algorithm: {algorithm}")
                print(f"  Fail Rate: {failed_bounds / len(dataframe) * 100:.2f}%")
                if failed_bounds > 0:
                    print(f"  Invalid Rate: {invalid_bounds / failed_bounds * 100:.2f}%")
                else:
                    print(f"  Invalid Rate: {invalid_bounds / len(dataframe) * 100:.2f}%")

                print(f"  Net Bound Width: {without__failed_and_invalid[f'{algorithm}_bound_width'].mean()}")
            else:
                print(f"Algorithm: {algorithm} not found in dataframe columns.")

    @staticmethod
    def plot_smoothed_ate_vs_bounds(dataframe, algorithms=['autobound'], window=30):
        """
        Plot smoothed ATE_true and confidence intervals for multiple algorithms from the given dataframe.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing columns 'ATE_true', '<algorithm>_bound_lower', '<algorithm>_bound_upper', and 'b_X_Y'.
        algorithms (list): List of algorithm names to use for bounds (e.g., ['autobound', 'causaloptim']).
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

        # Plot the smoothed data
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='b_X_Y', y='ATE_true_smooth', label='$ATE_{true}$', color='blue')

        for algorithm in algorithms:
            if f'{algorithm}_bound_lower' in df.columns and f'{algorithm}_bound_upper' in df.columns:
                df[f'{algorithm}_bound_lower_smooth'] = df[f'{algorithm}_bound_lower'].rolling(window=window, center=True).mean()
                df[f'{algorithm}_bound_upper_smooth'] = df[f'{algorithm}_bound_upper'].rolling(window=window, center=True).mean()
                sns.lineplot(data=df, x='b_X_Y', y=f'{algorithm}_bound_lower_smooth', color=f'C{algorithms.index(algorithm)}')
                sns.lineplot(data=df, x='b_X_Y', y=f'{algorithm}_bound_upper_smooth', color=f'C{algorithms.index(algorithm)}', label=f'{algorithm} Bounds')
            else:
                print(f"Warning: Columns for algorithm '{algorithm}' not found in dataframe.")

        plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.title('Algorithms vs $ATE_{true}$ (smoothed out)')
        plt.xlabel('b_X_Y Coefficient')
        plt.ylabel('ATE Value')
        plt.legend()
        plt.grid(True)
        plt.show()