# contains dataset metadata, such as columns that need one-hot encoding, label identifiers, etc.

from .datasets import ds_meta
from .preprocess import download_datasets, preprocess_datasets
from .experiments import run_experiments
from .plot import plot_results


def run_UCI_MLR():

    print('-'*52)
    print('Experiments on UCI Machine Learning Repository Data')
    print('-'*52)
    
    # Download all datasets
    download_datasets(ds_meta) # Comment out this line if data sets have previously been downloaded
    preprocess_datasets(ds_meta)

    # Evaluate practicality and applicability of the ROC Convex Hull Method
    run_experiments(ds_meta)

    # Plot results from experiments
    plot_results()