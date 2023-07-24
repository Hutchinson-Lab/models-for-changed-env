# Main script for running all experiments

# Nahian Ahmed
# July 23, 2023

from Simulated.run_Simulated import run_Simulated
from UCI_MLR.run_UCI_MLR import run_UCI_MLR

# Experiments on Simulated Datasets
run_Simulated()

# Experiments on UCI Machine Learning Repository Datasets
run_UCI_MLR()
