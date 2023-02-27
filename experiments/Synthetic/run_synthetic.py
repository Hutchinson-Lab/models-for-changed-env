# contains dataset metadata, such as columns that need one-hot encoding, label identifiers, etc.
from experiments.Synthetic.datasets import ds_meta
from experiments.Synthetic.preprocess import generate_datasets, preprocess_datasets
from experiments.Synthetic.experiments import run_experiments
from experiments.Synthetic.plot import plot_results

def run_synthetic():
    # Generate all datasets
    generate_datasets(ds_meta) # Comment out this line if data sets have previously been downloaded
    preprocess_datasets(ds_meta)

    # Evaluate practicality and applicability of the ROC Convex Hull Method
    run_experiments(ds_meta)

    # Plot results from experiments
    plot_results()