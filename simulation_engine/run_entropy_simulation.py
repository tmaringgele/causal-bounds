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
    from simulation_engine.scenarios.conf.binary_entropy_conf import BinaryEntropyConf
    #install the R causaloptim package
    # install_causaloptim()
    r('.libPaths(c("'+R_path+'/site-library", .libPaths()))')

    print(f"Running simulation with N_simulations = {N_simulations}", flush=True)
    print(f"So each H_target has N_simulations/10 = {N_simulations/10}", flush=True)    
    
    
    h_targets = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    results = {}

    for h in h_targets:
        data = BinaryEntropyConf.generate_data_rolling_ate(int(N_simulations/10), uniform_confounder_entropy=True, noise=False, h_target=h)
        scenario = BinaryEntropyConf(data)
        scenario.run()
        scenario.data
        results[h] = scenario.data

    results_df = pd.concat(results.values(), ignore_index=True)

    runtimes = scenario.run()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([runtimes['runtimes']]).to_csv(f'runtimes_{timestamp}.csv', index=False)
    results.to_pickle(f'results_array_{timestamp}.pkl')
    results_df.to_pickle(f'results_df_{timestamp}.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("N_simulations", type=int, help="Number of simulations to run")
    parser.add_argument("--R_path", type=str, default="D:/Program Files/R/R-4.3.1", help="Path to R installation")
    ## Example usage: python .\run_entropy_simulation.py 2 --R_path "D:/Program Files/R-4.5.0"
    ## Example usage: python .\run_entropy_simulation.py 2 --R_path "/usr/lib/R"

    args = parser.parse_args()
    main(args.N_simulations, args.R_path)