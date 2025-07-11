# Causal Bounds

This repository contains the implementation and experiments for my Bachelor's Thesis: **"Bounding Causal Effects and Counterfactuals"**.

## 🚀 Quick Start

### Running Algorithms

For a comfortable way to run the algorithms, I recommend using my Python package:

**[CausalBoundingEngine](https://github.com/tmaringgele/CausalBoundingEngine)**

### Recreating Thesis Experiments

To reproduce the experiments from my thesis, follow these steps:

#### Prerequisites
- Python with all required dependencies
- R programming language
- Java

#### Running Simulations

1. **Core Scenarios**: Use the main simulation script
   ```bash
   python run_all_simulations.py 2000 --R_path "/usr/lib/R"
   ```

2. **Binary Entropy Confounding**: Use the specialized entropy simulation
   ```bash
   python run_entropy_simulation.py 2000 --R_path "/usr/lib/R"
   ```

Both scripts are located in the `simulation_engine` folder.

**Parameters:**
- First argument: Number of simulations (I used 2000 for thesis results)
- `--R_path`: Path to your R installation

#### Analyzing Results

To generate bound statistics tables, use:
```python
from simulation_engine.util.plotting_util import print_bound_statistics_table
print_bound_statistics_table()
```

## 📁 Repository Structure

- `simulation_engine/`: Main simulation scripts and utilities
- Other folders: Experimental code and result analysis tools

## 📊 Results

The simulation results and analysis tools are included in various folders throughout the repository for exploring algorithm performance and bound quality.



## 📚 References

This repository implements algorithms from the following research works:

- **Manski (1990)** - Nonparametric bounds foundation
- **Tian & Pearl (2000)** - Probability of causation bounds  
- **Duarte et al. (2023)** - Autobound optimization approach
- **Jiang & Shpitser (2020)** - Entropy-based weak confounding
- **Sachs et al. (2022)** - Causaloptim R library
- **Zaffalon et al. (2022)** - Causal expectation maximisation approach
- **Zhang & Bareinboim (2021)** - Continuous outcome bounding

For detailed references and citations, please consult:
- The thesis PDF document
- [CausalBoundingEngine documentation](https://github.com/tmaringgele/CausalBoundingEngine)