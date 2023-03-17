# contains dataset metadata, such as columns that need one-hot encoding, label identifiers, etc.
from .datasets import ds_meta
from .preprocess import generate_datasets
from .experiments import run_experiments
from .plot import plot_results

def run_Synthetic():
    
    print('-'*34)
    print('Experiments on Synthetic Data Sets')
    print('-'*34)

    # Generate all datasets
    generate_datasets(ds_meta) # Comment out this line if data sets have previously been generated
 
    # Evaluate practicality and applicability of the ROC Convex Hull Method
    run_experiments(ds_meta)

    # Plot results from experiments
    plot_results()