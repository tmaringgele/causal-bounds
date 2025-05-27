from abc import ABC, abstractmethod
import time
from datetime import datetime

class IVScenario(ABC):
    def __init__(self, dag):
        self.dag = dag
        self.data = None

    AVAILABLE_ALGORITHMS = {}

    def run_all_bounding_algorithms(self, algorithms=None):
        """
        Run all bounding algorithms, print runtime statistics, and return the runtimes and current timestamp.

        Args:
            algorithms (list, optional): List of algorithm names to run. If None, all algorithms are run.

        Returns:
            dict: A dictionary with algorithm runtimes and the current timestamp.
                    Example: {"runtimes": {"2SLS": 1.23, "manski": 0.45}, "timestamp": "2023-03-15T12:34:56"}
        """
        available_algorithms = self.AVAILABLE_ALGORITHMS

        if algorithms is None:
            algorithms = available_algorithms.keys()

        runtimes = {}
        total_start_time = time.time()

        for algo in algorithms:
            if algo in available_algorithms:
                print(f"Running {algo}...")
                start_time = time.time()
                available_algorithms[algo](self)
                end_time = time.time()
                runtime = end_time - start_time
                runtimes[algo] = runtime
                print(f"{algo} completed in {runtime:.2f} seconds.")
            else:
                print(f"Algorithm '{algo}' is not recognized.")

        total_end_time = time.time()
        total_runtime = total_end_time - total_start_time
        print(f"Total runtime: {total_runtime:.2f} seconds.")

        current_timestamp = datetime.now().isoformat()
        return {"runtimes": runtimes, "timestamp": current_timestamp}

