from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import pandas as pd
from simulation_engine.util.alg_util import AlgUtil

class AutoBound:
    """
    Class to run the AutoBound algorithm for causal inference.
    """

    @staticmethod 
    def bound_binaryIV(query, data, dagstring="Z -> X, X -> Y, U -> X, U -> Y", unob="U"):
        for idx, sim in data.iterrows():
            df = pd.DataFrame({'Y': sim['Y'], 'X': sim['X'], 'Z': sim['Z']})
            failed = False
            try:
                joint_probs = AutoBound._compute_joint_probabilities_IV(df)
                bound_lower, bound_upper = AutoBound.run_experiment_binaryIV(query, dagstring, unob, joint_probs)

            except Exception as e:
                print(f"Error in AutoBound: {e}")
                failed = True
            
            #Flatten bounds to trivial ceils
            if failed | (bound_upper > AlgUtil.get_trivial_Ceils(query)[1]):
                bound_upper = AlgUtil.get_trivial_Ceils(query)[1] 
            if failed | (bound_lower < AlgUtil.get_trivial_Ceils(query)[0]): 
                bound_lower = AlgUtil.get_trivial_Ceils(query)[0]

            bounds_valid = bound_lower <= sim[query+'_true'] <= bound_upper
            bounds_width = bound_upper - bound_lower

            data.at[idx, query+'_autobound_bound_lower'] = bound_lower
            data.at[idx, query+'_autobound_bound_upper'] = bound_upper
            data.at[idx, query+'_autobound_bound_valid'] = bounds_valid
            data.at[idx, query+'_autobound_bound_width'] = bounds_width
            data.at[idx, query+'_autobound_bound_failed'] = failed
        return data

    @staticmethod
    def run_experiment_binaryIV(query, dagstring, unob, joint_probs):
        """
        Run the AutoBound experiment.
        Parameters:
            dag (DAG): The directed acyclic graph representing the causal structure.
            df (pd.DataFrame): DataFrame containing the data for the experiment.
        Returns:
            tuple: (lower_bound, upper_bound) from AutoBound
        """
        dag = DAG()
        dag.from_structure(dagstring, unob)   
        
        problem = causalProblem(dag)

        problem.load_data_pandas(joint_probs)
        problem.add_prob_constraints()

        if query == 'ATE':
            problem.set_ate(ind='X', dep='Y')
        elif query == 'PNS':
            pns_query = problem.query('Y(X=1)=1 & Y(X=0)=0')
            problem.set_estimand(pns_query)
        else:
            raise ValueError("Query must be either 'ATE' or 'PNS'.")

        program = problem.write_program()
        lb, ub = program.run_pyomo(solver_name='glpk', verbose=False)

        return lb, ub
    
    @staticmethod
    def _compute_joint_probabilities_IV(df):
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