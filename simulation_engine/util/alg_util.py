
class AlgUtil:
    """
    Utility class for algorithms.
    """
    @staticmethod
    def get_trivial_Ceils(query):
        """
        Get trivial Ceils for the given query.

        Args:
            query (str): The query type (e.g., 'ATE' or 'PNS').

        Returns:
            tuple: A tuple containing the lower and upper bounds.
        """
        if query == 'ATE':
            return -1, 1
        elif query == 'PNS':
            return 0, 1
        else:
            raise ValueError(f"Unknown query type: {query}")
               
    @staticmethod
    def flatten_bounds_to_trivial_ceils(query, bound_lower, bound_upper, failed):
        #Flatten bounds to trivial ceils
        if failed | (bound_upper > AlgUtil.get_trivial_Ceils(query)[1]):
            bound_upper = AlgUtil.get_trivial_Ceils(query)[1] 
        if failed | (bound_lower < AlgUtil.get_trivial_Ceils(query)[0]): 
            bound_lower = AlgUtil.get_trivial_Ceils(query)[0]
        return bound_lower, bound_upper