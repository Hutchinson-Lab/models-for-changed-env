# contains dataset metadata, such as columns that need one-hot encoding, label identifiers, etc.
from experiments.UCI_MLR.datasets import ds_meta
from experiments.UCI_MLR.preprocess import download_datasets, preprocess_datasets
from experiments.UCI_MLR.experiments import run_experiments
from experiments.UCI_MLR.plot import plot_results


def run_UCI_MLR():
    # Download all datasets
    download_datasets(ds_meta) # Comment out this line if data sets have previously been downloaded
    preprocess_datasets(ds_meta)

    # Evaluate practicality and applicability of the ROC Convex Hull Method
    run_experiments(ds_meta)

    # Plot results from experiments
    plot_results()