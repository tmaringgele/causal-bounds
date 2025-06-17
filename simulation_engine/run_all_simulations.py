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

def run_binaryIV(N_simulations, foldername):
    from simulation_engine.scenarios.iv.binary_iv import BinaryIV
    print(f"Running BinaryIV simulation with N_simulations = {N_simulations}", flush=True)    
    
    data = BinaryIV.generate_data_rolling_ate(N_simulations)
    print("Data generation complete", flush=True)

    binaryIV = BinaryIV('IV Dag', data)

    runtimes = binaryIV.run()
    results = binaryIV.data

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([runtimes['runtimes']]).to_csv(f'{foldername}/runtimes_binaryIV_{timestamp}.csv', index=False)
    results.to_pickle(f'{foldername}/results_binaryIV_{timestamp}.pkl')

def run_contIV(N_simulations, foldername):
    from simulation_engine.scenarios.iv.continuous_iv import ContinuousIV
    print(f"Running ContIV simulation with N_simulations = {N_simulations}", flush=True)    
    
    data = ContinuousIV.run_rolling_b_X_Y_simulations(N_points=N_simulations, replications=1, n=500)
    print("Data generation complete", flush=True)

    contIV = ContinuousIV('IV Dag', data)

    runtimes = contIV.run()
    results = contIV.data

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([runtimes['runtimes']]).to_csv(f'{foldername}/runtimes_contIV_{timestamp}.csv', index=False)
    results.to_pickle(f'{foldername}/results_contIV_{timestamp}.pkl')

def run_binaryConf(N_simulations, foldername):
    from simulation_engine.scenarios.conf.binary_conf import BinaryConf
    print(f"Running binaryConf simulation with N_simulations = {N_simulations}", flush=True)    
    
    data = BinaryConf.generate_data_rolling_ate(N_simulations)
    print("Data generation complete", flush=True)

    binaryIV = BinaryConf('IV Dag', data)

    runtimes = binaryIV.run()
    results = binaryIV.data

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([runtimes['runtimes']]).to_csv(f'{foldername}/runtimes_binaryConf_{timestamp}.csv', index=False)
    results.to_pickle(f'{foldername}/results_binaryConf_{timestamp}.pkl')

def run_contConf(N_simulations, foldername):
    from simulation_engine.scenarios.conf.continuous_conf import ContinuousConf
    print(f"Running contConf simulation with N_simulations = {N_simulations}", flush=True)    
    
    data = ContinuousConf.generate_data_rolling_ate(N_simulations)
    print("Data generation complete", flush=True)

    binaryIV = ContinuousConf('IV Dag', data)

    runtimes = binaryIV.run()
    results = binaryIV.data

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([runtimes['runtimes']]).to_csv(f'{foldername}/runtimes_contConf_{timestamp}.csv', index=False)
    results.to_pickle(f'{foldername}/results_contConf_{timestamp}.pkl')
    


def main(N_simulations, R_path):
    print(f"Setting R path to {R_path}", flush=True)
    os.environ['R_HOME'] = R_path
    #install the R causaloptim package
    # install_causaloptim()
    r('.libPaths(c("'+R_path+'/site-library", .libPaths()))')
    
    #use current timestamp for folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f'simulation_results_{timestamp}'
    os.makedirs(foldername, exist_ok=True)
    
    run_binaryIV(N_simulations, foldername)
    run_contIV(N_simulations, foldername)
    run_binaryConf(N_simulations, foldername)
    run_contConf(N_simulations, foldername)

    print(f"All simulations completed. Results saved in {foldername}")
    print(f"Total Runtime: {datetime.datetime.now() - datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("N_simulations", type=int, help="Number of simulations to run")
    parser.add_argument("--R_path", type=str, default="D:/Program Files/R/R-4.3.1", help="Path to R installation")
    ## Example usage: python .\run_all_simulation.py 2 --R_path "D:/Program Files/R-4.5.0"
    ## Example usage: python run_all_simulations.py 2 --R_path "/usr/lib/R"

    args = parser.parse_args()
    main(args.N_simulations, args.R_path)