'''

'''
from datasets import ds_meta # contains dataset metadata, such as columns that need one-hot encoding, label identifiers, etc.
from preprocess import download_datasets 
from experiments import run_experiments
from plot import plot_results

# Experiments on synthetic data


# Download all datasets
download_datasets(ds_meta) # Comment out this line if data sets have previously been downloaded

# Evaluate practicality and applicability of the ROC Convex Hull Method
run_experiments()

# Plot results from experiments
plot_results()