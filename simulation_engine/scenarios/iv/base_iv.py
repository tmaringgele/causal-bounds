from abc import ABC, abstractmethod

class IVScenario(ABC):
    def __init__(self, dag):
        self.dag = dag
        self.data = None
