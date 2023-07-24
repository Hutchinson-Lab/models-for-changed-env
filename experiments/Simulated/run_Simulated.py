# Main script for running experiments on simulated datasets

# Nahian Ahmed
# July 23, 2023

# contains dataset metadata, such as columns that need one-hot encoding, label identifiers, etc.
from .datasets import ds_meta
from .preprocess import generate_datasets
from .experiments import run_experiments
from .plot import plot_results

def run_Simulated():
    
    print('\n')
    print('-'*33)
    print('Experiments on Simulated Datasets')
    print('-'*33)

    # Generate all datasets
    generate_datasets(ds_meta) # Comment out this line if datasets have previously been generated
 
    # Evaluate practicality and applicability of the ROC Convex Hull Method
    run_experiments(ds_meta)

    # Plot results from experiments
    plot_results()