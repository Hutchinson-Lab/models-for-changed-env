# Main script for running all experiments

# Nahian Ahmed
# July 23, 2023

from Synthetic.run_Synthetic import run_Synthetic
from UCI_MLR.run_UCI_MLR import run_UCI_MLR

# Experiments on Synthetic Datasets
run_Synthetic()

# Experiments on UCI Machine Learning Repository Datasets
run_UCI_MLR()
