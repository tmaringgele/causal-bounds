from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import pandas as pd

class AutoBound:
    """
    Class to run the AutoBound algorithm for causal inference.
    """

    @staticmethod
    def run_experiment_binaryIV_ATE(df):
        """
        Run the AutoBound experiment.
        Parameters:
            dag (DAG): The directed acyclic graph representing the causal structure.
            df (pd.DataFrame): DataFrame containing the data for the experiment.
        Returns:
            tuple: (lower_bound, upper_bound) from AutoBound
        """
        dag = DAG()
        dag.from_structure("Z -> X, X -> Y, U -> X, U -> Y", unob = "U")   
        joint_probs = AutoBound._compute_joint_probabilities(pd.DataFrame({'X': df['X'], 'Y': df['Y'], 'Z': df['Z']}))
        
        problem = causalProblem(dag)

        problem.load_data_pandas(joint_probs)
        problem.add_prob_constraints()
        problem.set_ate(ind='X', dep='Y')

        program = problem.write_program()
        lb, ub = program.run_pyomo(solver_name='glpk', verbose=False)

        return lb, ub
    
    @staticmethod
    def _compute_joint_probabilities(df):
        """
        Computes the joint probabilities for each combination of Z, X, and Y in the input DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame with columns ['X', 'Y', 'Z'].

        Returns:
            pd.DataFrame: DataFrame with columns ['Z', 'X', 'Y', 'prob'] representing the joint probabilities.
        """
        # Count occurrences of each combination of Z, X, Y
        joint_counts = df.groupby(['Z', 'X', 'Y']).size().reset_index(name='count')
        
        # Calculate total number of rows in the input DataFrame
        total_count = len(df)
        
        # Compute probabilities
        joint_counts['prob'] = joint_counts['count'] / total_count
        
        # Drop the count column as it's not needed in the output
        joint_probs = joint_counts.drop(columns=['count'])
        
        return joint_probs