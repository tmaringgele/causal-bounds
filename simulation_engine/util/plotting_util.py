import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from simulation_engine.util.alg_util import AlgUtil

class PlottingUtil:
    """
    A utility class for plotting and visualization.
    """

    @staticmethod
    def print_bound_statistics(dataframe, algorithms=['autobound', 'causaloptim'], query=None):
        """
        Print statistics of the bounds for the given algorithms.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing bound information.
        algorithms (list): List of algorithm names to print statistics for.
        query (str, optional): If set, only algorithms starting with this string are printed.

        Returns:
        None
        """
        # Filter algorithms if query is set
        if query is not None:
            filtered_algorithms = [alg for alg in algorithms if alg.startswith(query)]
        else:
            filtered_algorithms = algorithms

        for algorithm in filtered_algorithms:
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
    def print_bound_statistics_table(dataframe, algorithms=['autobound', 'causaloptim'], query=None):
        """
        Print statistics of the bounds for the given algorithms in a table format.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing bound information.
        algorithms (list): List of algorithm names to print statistics for.
        query (str, optional): If set, only algorithms starting with this string are printed.

        Returns:
        None
        """
        # Filter algorithms if query is set
        if query is not None:
            filtered_algorithms = [alg for alg in algorithms if alg.startswith(query)]
        else:
            filtered_algorithms = algorithms

        # Work on a copy to avoid modifying the original dataframe
        df = dataframe.copy()
        
        stats = []
        for algorithm in filtered_algorithms:
            if f'{algorithm}_bound_valid' in df.columns:
                query = algorithm.split('_')[0]  # Extract query type from algorithm name

                # Ensure bound_valid is correct
                # it should be True iff lower <= true <= upper
                df[f'{algorithm}_bound_valid'] = (df[f'{algorithm}_bound_lower'] <= df[f'{query}_true']) & \
                                                (df[f'{algorithm}_bound_upper'] >= df[f'{query}_true'])
                
                

                failed_bounds = df[df[f'{algorithm}_bound_failed']].shape[0]
                without_failed = df[df[f'{algorithm}_bound_failed'] == False]
                invalid_bounds = without_failed[without_failed[f'{algorithm}_bound_valid'] == False].shape[0]
                without_failed_and_invalid = without_failed[without_failed[f'{algorithm}_bound_valid'] == True]
                fail_rate = failed_bounds / len(df) * 100
                invalid_rate = invalid_bounds / (len(df) - failed_bounds) * 100 if (len(df) - failed_bounds) > 0 else np.nan

                # --- Compute Avg. Invalid dist. (as % of query range) ---
                invalid_df = without_failed[without_failed[f'{algorithm}_bound_valid'] == False]
                lower_col = f'{algorithm}_bound_lower'
                upper_col = f'{algorithm}_bound_upper'
                true_col = f'{query}_true'
                if not invalid_df.empty and true_col in df.columns:
                    lower = invalid_df[lower_col]
                    upper = invalid_df[upper_col]
                    true = invalid_df[true_col]
                    dist = np.where(true < lower, (lower - true).abs(),
                                    np.where(true > upper, (true - upper).abs(), 0))
                    # Determine query range
                    if query.upper() == "ATE":
                        query_range = 2.0
                    elif query.upper() == "PNS":
                        query_range = 1.0
                    else:
                        query_range = 1.0  # Default to 1 if unknown
                    avg_invalid_dist = np.mean(dist) / query_range if len(dist) > 0 else None
                else:
                    avg_invalid_dist = None

                # --- Set trivial ceils on the copy only ---
                failed_mask = df[f'{algorithm}_bound_failed'] == True
                invalid_mask = (df[f'{algorithm}_bound_failed'] == False) & (df[f'{algorithm}_bound_valid'] == False)
                df.loc[failed_mask | invalid_mask, lower_col] = AlgUtil.get_trivial_Ceils(query)[0]
                df.loc[failed_mask | invalid_mask, upper_col] = AlgUtil.get_trivial_Ceils(query)[1]
                df[f'{algorithm}_bound_width'] = df[upper_col] - df[lower_col]

                bound_width = df[f'{algorithm}_bound_width'].mean()
                net_bound_width = without_failed_and_invalid[f'{algorithm}_bound_width'].mean()

                stats.append({
                    'Algorithm': algorithm,
                    'Fail Rate (%)': f"{fail_rate:.2f}",
                    'Invalid Rate (%)': f"{invalid_rate:.2f}",
                    'Net Bound Width': f"{net_bound_width:.4f}" if net_bound_width is not None else "N/A",
                    'Bound Width': f"{bound_width:.4f}" if bound_width is not None else "N/A",
                    'Invalid Δ (%)': f"{(avg_invalid_dist * 100):.2f}" if avg_invalid_dist is not None else "N/A"
                })
            else:
                stats.append({
                    'Algorithm': algorithm,
                    'Fail Rate (%)': "N/A",
                    'Invalid Rate (%)': "N/A",
                    'Net Bound Width': "N/A",
                    'Avg. Invalid dist.': "N/A"
                })
        stats_df = pd.DataFrame(stats)
        # Sort by Net Bound Width (convert to float, "N/A" as NaN)
        stats_df['Net Bound Width Sort'] = pd.to_numeric(stats_df['Net Bound Width'], errors='coerce')
        stats_df = stats_df.sort_values(by='Net Bound Width Sort', ascending=True).drop(columns=['Net Bound Width Sort'])
        print(stats_df.to_string(index=False))

    @staticmethod
    def compute_tightest_bound(dataframe):
        """
        Compute the tightest bounds for each query type (ATE and PNS).
        
        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing bound information for different algorithms.
        
        Returns:
        pd.DataFrame: The same dataframe with two additional columns: 'PNS_tightest_bound' and 'ATE_tightest_bound'.
        """
        df = dataframe.copy()
        
        # Process each query type (ATE and PNS)
        for query in ['ATE', 'PNS']:
            # Get all columns containing bound_width for this query type
            bound_width_cols = [col for col in df.columns if f'{query}_' in col and col.endswith('_bound_width')]
            
            # Extract algorithm names from the column names
            algorithms = [col.replace(f'{query}_', '').replace('_bound_width', '') for col in bound_width_cols]
            
            # Create a new column for the tightest bound
            tightest_bound_col = f'{query}_tightest_bound'
            df[tightest_bound_col] = None
            
            # For each row, find the algorithm with the smallest bound width
            for idx, row in df.iterrows():
                valid_bounds = {}
                for alg in algorithms:
                    # Only check if the value is not NaN (don't check failed or validity)
                    if not pd.isna(row[f'{query}_{alg}_bound_width']):
                        valid_bounds[alg] = row[f'{query}_{alg}_bound_width']
                
                # If there are any bounds, find the one with minimum width
                if valid_bounds:
                    min_alg = min(valid_bounds, key=valid_bounds.get)
                    df.at[idx, tightest_bound_col] = min_alg
        
        return df

    @staticmethod
    def compute_tightest_bound_valid(dataframe):
        """
        Compute the tightest valid bounds for each query type (ATE and PNS).
        Only considers bounds where _bound_valid is true and _bound_failed is false.
        
        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing bound information for different algorithms.
        
        Returns:
        pd.DataFrame: The same dataframe with two additional columns: 'PNS_tightest_valid_bound' and 'ATE_tightest_valid_bound'.
        """
        df = dataframe.copy()
        
        # Process each query type (ATE and PNS)
        for query in ['ATE', 'PNS']:
            # Get all columns containing bound_width for this query type
            bound_width_cols = [col for col in df.columns if f'{query}_' in col and col.endswith('_bound_width')]
            
            # Extract algorithm names from the column names
            algorithms = [col.replace(f'{query}_', '').replace('_bound_width', '') for col in bound_width_cols]
            print(f"{query} algorithms: {algorithms}")
            # Create a new column for the tightest valid bound
            tightest_bound_col = f'{query}_tightest_bound_valid'
            df[tightest_bound_col] = None
            
            # For each row, find the algorithm with the smallest valid bound width
            for idx, row in df.iterrows():
                valid_bounds = {}
                for alg in algorithms:
                    # Check if the bound is valid and not failed
                    if (f'{query}_{alg}_bound_valid' in df.columns and 
                        f'{query}_{alg}_bound_failed' in df.columns and
                        row[f'{query}_{alg}_bound_valid'] == True and
                        row[f'{query}_{alg}_bound_failed'] == False and
                        not pd.isna(row[f'{query}_{alg}_bound_width'])):
                        valid_bounds[alg] = row[f'{query}_{alg}_bound_width']
                
                # If there are valid bounds, find the one with minimum width
                if valid_bounds:
                    min_alg = min(valid_bounds, key=valid_bounds.get)
                    df.at[idx, tightest_bound_col] = min_alg
        
        return df

    @staticmethod
    def plot_tightest_bounds_distribution(dataframe, valid_only=False, figsize=(12, 6)):
        """
        Plot the distribution of algorithms that provided the tightest bounds.
        
        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing tightest bound information.
        valid_only (bool): If True, use only valid bounds (from `_tightest_bound_valid`), otherwise use all bounds.
        figsize (tuple): Size of the figure (width, height).
        
        Returns:
        None
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Select the appropriate column suffix based on valid_only parameter
        column_suffix = "tightest_bound_valid" if valid_only else "tightest_bound"
        
        # Plot distribution for ATE and PNS
        for i, query in enumerate(['ATE', 'PNS']):
            col_name = f"{query}_{column_suffix}"
            
            if col_name not in dataframe.columns:
                print(f"Column {col_name} not found in dataframe")
                continue
            
            # Count occurrences of each algorithm
            counts = dataframe[col_name].value_counts().sort_values(ascending=False)
            
            # Calculate percentages
            total = counts.sum()
            percentages = (counts / total * 100).round(1)
            
            # Create labels with percentages
            labels = [f"{alg}\n({pct}%)" for alg, pct in zip(counts.index, percentages)]
            
            # Plot bar chart
            ax = axes[i]
            bars = ax.bar(counts.index, counts.values, color='skyblue', edgecolor='black')
            
            # Add percentage labels on top of bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                        f'{pct}%', ha='center', va='bottom', fontsize=9)
            
            # Set title and labels
            ax.set_title(f"{query} Tightest {'Valid ' if valid_only else ''}Bounds Distribution")
            ax.set_ylabel('Count')
            ax.set_xlabel('Algorithm')
            
            # Rotate x-axis labels if there are many algorithms
            if len(counts) > 5:
                ax.set_xticklabels(counts.index, rotation=45, ha='right')
                
            # Add grid lines for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Show counts on top of bars
            for i, v in enumerate(counts.values):
                ax.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_smoothed_query_vs_bounds(dataframe, query, algorithms=['autobound'], window=1, zeroline=False, roll_over='b_X_Y'):
        """
        Plot smoothed <query>_true and confidence intervals for multiple algorithms from the given dataframe.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing columns '<query>_true', '<algorithm>_bound_lower', '<algorithm>_bound_upper', and the roll_over variable.
        query (str): The query type (e.g., 'ATE' or 'PNS') to plot.
        algorithms (list): List of algorithm names to use for bounds (e.g., ['autobound', 'causaloptim']).
        window (int): The size of the rolling window for smoothing. Default is 30.
        zeroline (bool): Whether to plot a horizontal zero line.
        roll_over (str): The column name to use for the X axis. Default is 'b_X_Y'.

        Returns:
        None
        """
        # Create a copy of the dataframe to avoid modifying the original
        df = dataframe.copy()

        # Check if roll_over column has varying values
        if roll_over not in df.columns or df[roll_over].nunique() <= 1:
            print(f"Error: The '{roll_over}' column has constant or invalid values. Cannot plot.")
            return

        # Smoothen the data using a rolling average
        df[f'{query}_true_smooth'] = df[f'{query}_true'].rolling(window=window, center=True).mean()

        # Plot the smoothed data
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=roll_over, y=f'{query}_true_smooth', label=f'${query}_{{true}}$', color='blue')

        alpha = 0.8
        for algorithm in algorithms:
            if f'{algorithm}_bound_lower' in df.columns and f'{algorithm}_bound_upper' in df.columns:
                df[f'{algorithm}_bound_lower_smooth'] = df[f'{algorithm}_bound_lower'].rolling(window=window, center=True).mean()
                df[f'{algorithm}_bound_upper_smooth'] = df[f'{algorithm}_bound_upper'].rolling(window=window, center=True).mean()
                sns.lineplot(data=df, x=roll_over, y=f'{algorithm}_bound_lower_smooth', color=f'C{algorithms.index(algorithm)}', alpha=alpha)
                sns.lineplot(data=df, x=roll_over, y=f'{algorithm}_bound_upper_smooth', color=f'C{algorithms.index(algorithm)}', label=f'{algorithm}', alpha=alpha)
            else:
                print(f"Warning: Columns for algorithm '{algorithm}' not found in dataframe.")

        if zeroline:
            plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.title(f'Algorithms vs ${query}_{{true}}$ (smoothed out)')
        plt.xlabel(roll_over)
        plt.ylabel(f'{query} Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_trueEntropyUB_vs_bound_width(dataframe, entropybound_prefix='entropybounds-trueTheta'):
        """
        Plot the true entropy upper bound against the bound width.

        Parameters:
        dataframe (pd.DataFrame): The input dataframe containing entropy bounds with true theta.
        entropybound_prefix (str): The prefix for the entropy bounds columns.

        Returns:
        None
        """
        #plot theta on x-axis and bound width on y-axis
        plt.figure(figsize=(8,6))
        plt.scatter(dataframe[f'{entropybound_prefix}_theta'], dataframe[f'{entropybound_prefix}_bound_width'], alpha=0.5, label='Valid Bounds')
        plt.scatter(dataframe[dataframe[f'{entropybound_prefix}_bound_valid'] == False][f'{entropybound_prefix}_theta'],
                    dataframe[dataframe[f'{entropybound_prefix}_bound_valid'] == False][f'{entropybound_prefix}_bound_width'],
                    color='red', alpha=0.5, label='Invalid Bounds')
        plt.xlabel(r'entropy(U) = θ')
        plt.ylabel('Bound Width')
        plt.title(r'θ vs Bound Width. True Theta')
        plt.grid(True)
        plt.legend()
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





        df_ols['theta'] = df_ols[f'{entropybound_prefix}_theta']
        df_ols['log_theta'] = np.log(df_ols['theta'] + 1e-5)  # Add small value to avoid log(0)

        # Fix: patsy/statsmodels does not allow '-' in variable names, so use a temporary column name
        bound_width_col = f'{entropybound_prefix}_bound_width'
        safe_bound_width_col = 'bound_width_tmp'
        df_ols[safe_bound_width_col] = df_ols[bound_width_col]

        # Fractional Logit Model with log(theta)
        model = smf.glm(formula=f"{safe_bound_width_col} ~ theta + log_theta",
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
        plt.scatter(dataframe[f'{entropybound_prefix}_theta'], dataframe[f'{entropybound_prefix}_bound_width'], alpha=0.5, label='Valid Bounds')
        plt.scatter(dataframe[dataframe[f'{entropybound_prefix}_bound_valid'] == False][f'{entropybound_prefix}_theta'],
                    dataframe[dataframe[f'{entropybound_prefix}_bound_valid'] == False][f'{entropybound_prefix}_bound_width'],
                    color='red', alpha=0.5, label='Invalid Bounds')
        # plt.axvline(actual_entropy, color='blue', linestyle='--', label='Actual Entropy')
        # plt.axvline(0.1, color='purple', linestyle='--', label='θ ∈ {0.05, 0.1}')
        # plt.axvline(0.05, color='purple', linestyle='--')

        plt.plot(theta_pred, y_pred, color='green', linewidth=2, label='Fractional Logit Fit')
        plt.xlabel(r'θ (upper bound for the entropy of U)')
        plt.ylabel('Bound Width')
        plt.title(r'θ vs Bound Width')
        plt.grid(True)
        plt.legend()
        plt.show()
    @staticmethod
    def plot_ate_pns(df: pd.DataFrame, x_col: str = "b_X_Y", ate_col: str = "ATE_true", pns_col: str = "PNS_true", window: int = 1):
        """
        Plot ATE and PNS from a DataFrame of simulation results, with optional smoothing.

        Args:
            df (pd.DataFrame): DataFrame containing simulation results.
            x_col (str): Column name representing the rolling parameter (e.g. b_X_Y).
            ate_col (str): Column name for the true ATE values.
            pns_col (str): Column name for the true PNS values.
            window (int): Rolling window size for smoothing. Default is 1 (no smoothing).
        """
        ate_smooth = df[ate_col].rolling(window=window, center=True).mean()
        pns_smooth = df[pns_col].rolling(window=window, center=True).mean()
        plt.figure(figsize=(8, 5))
        plt.plot(df[x_col], ate_smooth, label="ATE", linewidth=2)
        plt.plot(df[x_col], pns_smooth, label="PNS", linewidth=2)
        plt.xlabel(f"{x_col}")
        plt.ylabel("Value")
        plt.title("True ATE and PNS across simulations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()