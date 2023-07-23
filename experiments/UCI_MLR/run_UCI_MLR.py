# Main script for running experiments on UCI datasets

# Nahian Ahmed
# July 23, 2023

# contains dataset metadata, such as columns that need one-hot encoding, label identifiers, etc.
from .datasets import ds_meta
from .preprocess import download_datasets, preprocess_datasets
from .experiments import run_experiments
from .plot import plot_results


def run_UCI_MLR():

    print('\n')
    print('-'*55)
    print('Experiments on UCI Machine Learning Repository Datasets')
    print('-'*55)
    
    # Download all datasets
    download_datasets(ds_meta) # Comment out this line if datasets have previously been downloaded
    preprocess_datasets(ds_meta)

    # Evaluate practicality and applicability of the ROC Convex Hull Method
    run_experiments(ds_meta)

    # Plot results from experiments
    plot_results()