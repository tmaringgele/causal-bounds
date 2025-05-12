import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

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

                print(f"  Invalid Rate: {invalid_bounds / (len(dataframe) - failed_bounds) * 100:.2f}%")

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

        alpha = 0.8
        for algorithm in algorithms:
            if f'{algorithm}_bound_lower' in df.columns and f'{algorithm}_bound_upper' in df.columns:
                df[f'{algorithm}_bound_lower_smooth'] = df[f'{algorithm}_bound_lower'].rolling(window=window, center=True).mean()
                df[f'{algorithm}_bound_upper_smooth'] = df[f'{algorithm}_bound_upper'].rolling(window=window, center=True).mean()
                sns.lineplot(data=df, x='b_X_Y', y=f'{algorithm}_bound_lower_smooth', color=f'C{algorithms.index(algorithm)}', alpha=alpha)
                sns.lineplot(data=df, x='b_X_Y', y=f'{algorithm}_bound_upper_smooth', color=f'C{algorithms.index(algorithm)}', label=f'{algorithm}', alpha=alpha)
            else:
                print(f"Warning: Columns for algorithm '{algorithm}' not found in dataframe.")

        plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.title('Algorithms vs $ATE_{true}$ (smoothed out)')
        plt.xlabel('b_X_Y Coefficient')
        plt.ylabel('ATE Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def randomized_entropyUB_vs_bound_width(dataframe, entropybound_prefix='entropybounds'):
        """
        Plot the randomized entropy upper bound against the bound width.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing entropy bounds with RANDOMIZED theta. 
        entropybound_prefix (str): The prefix for the entropy bounds columns.

        Returns:
        None
        """
        
        df_ols = dataframe[
            (dataframe[f'{entropybound_prefix}_bound_valid']) & 
            (dataframe[f'{entropybound_prefix}_bound_failed'] == False)
        ].copy()

        df_ols['theta'] = df_ols[f'{entropybound_prefix}_H(conf)_UB']
        df_ols['log_theta'] = np.log(df_ols['theta'] + 1e-5)  # Add small value to avoid log(0)

        # Fractional Logit Model with log(theta)
        model = smf.glm(formula=f"{entropybound_prefix}_bound_width ~ theta + log_theta",
                        data=df_ols,
                        family=sm.families.Binomial(link=sm.families.links.logit()))
        result = model.fit()


        print(result.summary())

        # Create prediction grid
        theta_pred = np.linspace(df_ols['theta'].min(), df_ols['theta'].max(), 300)
        X_pred = pd.DataFrame({'theta': theta_pred, 'log_theta': np.log(theta_pred + 1e-5)})

        # Predict fitted values
        y_pred = result.predict(X_pred)

        # Calculate actual entropy 
        actual_entropy = df_ols['entropy_U'].mean()

        # Scatter plot + fitted curve
        plt.figure(figsize=(8,6))
        plt.scatter(dataframe[f'{entropybound_prefix}_H(conf)_UB'], dataframe[f'{entropybound_prefix}_bound_width'], alpha=0.5, label='Valid Bounds')
        plt.scatter(dataframe[dataframe[f'{entropybound_prefix}_bound_valid'] == False][f'{entropybound_prefix}_H(conf)_UB'],
                    dataframe[dataframe[f'{entropybound_prefix}_bound_valid'] == False][f'{entropybound_prefix}_bound_width'],
                    color='red', alpha=0.5, label='Invalid Bounds')
        # plt.axvline(actual_entropy, color='blue', linestyle='--', label='Actual Entropy')
        # plt.axvline(0.1, color='purple', linestyle='--', label='θ ∈ {0.05, 0.1}')
        # plt.axvline(0.05, color='purple', linestyle='--')

        # plt.plot(theta_pred, y_pred, color='green', linewidth=2, label='Fractional Logit Fit')
        plt.xlabel(r'θ (upper bound for the entropy of U)')
        plt.ylabel('Bound Width')
        plt.title(r'θ vs Bound Width')
        plt.grid(True)
        plt.legend()
        plt.show()