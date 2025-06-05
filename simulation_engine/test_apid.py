import argparse
import sys
sys.path.append("..")

from simulation_engine.scenarios.iv.continuous_iv import ContinuousIV
from simulation_engine.algorithms.apid import Apid
from simulation_engine.algorithms.apid_src.src.models.apid import APID

data = ContinuousIV.run_rolling_b_X_Y_simulations(
    b_range=(-5, 5), N_points=3, replications=1, n=500,
    # allowed_functions=['identity']
)

# Apid.run_from_generate_data(data.iloc[5])
results = Apid.bound_ATE(data, 'cuda')

results.to_csv("results/test_apid.csv", index=False)