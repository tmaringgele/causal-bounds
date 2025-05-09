from abc import ABC, abstractmethod

class IVScenario(ABC):
    def __init__(self, dag):
        self.dag = dag
        self.data = None

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def bound_ate_causaloptim(self, data_override=None):
        pass

    @abstractmethod
    def bound_pns_lp(self, data_override=None):
        pass
