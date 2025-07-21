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
    from simulation_engine.scenarios.conf.binary_conf import BinaryConf
    #install the R causaloptim package
    # install_causaloptim()
    r('.libPaths(c("'+R_path+'/site-library", .libPaths()))')

    print(f"Running simulation with N_simulations = {N_simulations}", flush=True)    
    
    simulations = []
    for i in range(N_simulations):
        sj_interest = BinaryConf._generate_data(n=500, b_U_X=1, b_U_Y=-1, b_X_Y=2, intercept_X=0.3, intercept_Y=-0.5, p_U=0.6, squasher_X_name='sigmoid', squasher_Y_name='sigmoid')
        simulations.append(sj_interest)
    data = pd.DataFrame(simulations)
    print("Data generation complete", flush=True)

    scenario = BinaryConf(data)

    runtimes = scenario.run()
    results = scenario.data

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([runtimes['runtimes']]).to_csv(f'runtimes_variance_{timestamp}.csv', index=False)
    results.to_pickle(f'results_variance_{timestamp}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("N_simulations", type=int, help="Number of simulations to run")
    parser.add_argument("--R_path", type=str, default="D:/Program Files/R/R-4.3.1", help="Path to R installation")
    ## Example usage: python .\run_variance_check.py 2 --R_path "D:/Program Files/R-4.5.0"
    ## Example usage: python .\run_variance_check.py 2 --R_path "/usr/lib/R"

    args = parser.parse_args()
    main(args.N_simulations, args.R_path)