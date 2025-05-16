import argparse
import sys
sys.path.append("..")
import datetime
import pandas as pd
import os
#disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import rpy2.robjects.packages as rpackages
import rpy2.robjects.vectors as rvectors
import rpy2.robjects as robjects

def install_causaloptim():
    # Ensure utils is available
    utils = rpackages.importr('utils')

    # Set a CRAN mirror
    utils.chooseCRANmirror(ind=1)  # index 1 = cloud.r-project.org

    # Install causaloptim
    utils.install_packages(rvectors.StrVector(['causaloptim']))


def main(N_simulations, R_path):
    print(f"Setting R path to {R_path}", flush=True)
    os.environ['R_HOME'] = R_path
    from simulation_engine.scenarios.iv.binary_iv import BinaryIV
    #install the R causaloptim package
    # install_causaloptim()

    print(f"Running simulation with N_simulations = {N_simulations}", flush=True)    
    
    data = BinaryIV.generate_data_rolling_ate(N_simulations)
    print("Data generation complete", flush=True)

    binaryIV = BinaryIV('IV Dag', data)

    runtimes = binaryIV.run_all_bounding_algorithms()
    results = binaryIV.data

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(runtimes).to_csv(f'runtimes_{timestamp}.csv', index=False)
    results.to_pickle(f'results_{timestamp}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("N_simulations", type=int, help="Number of simulations to run")
    parser.add_argument("--R_path", type=str, default="D:/Program Files/R/R-4.3.1", help="Path to R installation")
    ## Example usage: python .\run_simulation.py 2 --R_path "D:/Program Files/R-4.5.0"
    args = parser.parse_args()
    main(args.N_simulations, args.R_path)