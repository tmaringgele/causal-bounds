
from simulation_engine.scenarios.scenario import Scenario


class CustomDAG(Scenario):
    """
    A Scenario that can run various algorithms on a custom DAG.
    """

    AVAILABLE_ALGORITHMS = {
        "ATE_autobound": lambda self: AutoBound.bound_binaryIV("ATE", self.data, 
                        dagstring="X -> Y, U -> X, U -> Y",
                        unob="U",
                        ),
    }

    def __init__(self, dataframe, dagstring="X -> Y, U -> X, U -> Y", unobserved='U' ):
        super().__init__()
        self.data = dataframe
        self.dagstring = dag


        