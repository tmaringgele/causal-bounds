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
from rpy2.robjects import r

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
    from simulation_engine.scenarios.iv.continuous_iv import ContinuousIV
    #install the R causaloptim package
    # install_causaloptim()
    r('.libPaths(c("/usr/local/lib/R/site-library", .libPaths()))')

    print(f"Running simulation with N_simulations = {N_simulations}", flush=True)    
    
    data = ContinuousIV.run_rolling_b_X_Y_simulations( 
        b_range=(-10, 10), N_points=N_simulations, replications=1, n=500)
    print("Data generation complete", flush=True)

    contIV = ContinuousIV('IV Dag', data)

    runtimes = contIV.run()
    results = contIV.data

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([runtimes['runtimes']]).to_csv(f'runtimes_{timestamp}.csv', index=False)
    results.to_pickle(f'results_{timestamp}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("N_simulations", type=int, help="Number of simulations to run")
    parser.add_argument("--R_path", type=str, default="D:/Program Files/R/R-4.3.1", help="Path to R installation")
    ## Example usage: python .\run_cont_simulation.py 2 --R_path "D:/Program Files/R-4.5.0"
    ## Example usage: python .\run_cont_simulation.py 2 --R_path "/usr/lib/R"

    args = parser.parse_args()
    main(args.N_simulations, args.R_path)