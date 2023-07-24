# Generate and preprocess simulated datasets

# Nahian Ahmed
# July 23, 2023


import os, tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))


ds_main_dir = './experiments/Simulated/datasets/'
output_table_main_dir = './experiments/Simulated/tables/'
output_plot_main_dir = './experiments/Simulated/plots/'


def generate_datasets (ds_meta, theta = [1.0, 1.0, 1.0, 1.0], random_state=0):
    
    print("\nGenerating datasets:")
    
    if not os.path.isdir(ds_main_dir):
        os.makedirs(ds_main_dir)
    
    if not os.path.exists(output_table_main_dir):
        os.makedirs(output_table_main_dir)

    dataset_descriptions = pd.DataFrame(columns=('Dataset','Instances', 'Features', 'Categorical Features', 'Features After One-Hot', 'Class Balance', 'Added Noise'))
    i = 0
    

    

    for c in (pbar := tqdm.tqdm(ds_meta.keys())):
        pbar.set_description(f'Generating "{c}"')

        np.random.seed(random_state)

        n_cont_features = ds_meta[c]["n_features"] - ds_meta[c]["n_cat_features"]

        cont_features = None
        cat_features = None

        means = []
        if (n_cont_features != 0):
            cont_features = np.random.normal(size=(ds_meta[c]["n_instances"], n_cont_features))
            means.extend([0] * n_cont_features)
            if (n_cont_features == 1):
                cont_features = cont_features[:,np.newaxis]
            

        if (ds_meta[c]["n_cat_features"] != 0):
            cat_features = np.random.binomial(n=1, p=0.5, size=(ds_meta[c]["n_instances"], ds_meta[c]["n_cat_features"]))
            means.extend([0.5] * ds_meta[c]["n_cat_features"])
            if (ds_meta[c]["n_cat_features"] == 1):
                cat_features = cat_features[:,np.newaxis]

        X = None
        if (n_cont_features == 0):
            X = cat_features
        elif (ds_meta[c]["n_cat_features"] == 0):
            X = cont_features
        else:
            X = np.hstack((cont_features, cat_features))

                
        intercept = np.log(ds_meta[c]["class_distribution"]/(1-ds_meta[c]["class_distribution"])) - sum(means)

        i_ext = 0.1
        if (ds_meta[c]["class_distribution"] == 0.5):
            intercept += i_ext
        elif (ds_meta[c]["class_distribution"] == 0.25):
            intercept -= 6 * i_ext
        elif (ds_meta[c]["class_distribution"] == 0.75):
            intercept += 6 * i_ext
        
        z = intercept + theta[0]*X[:,0][:,np.newaxis] + theta[1]*X[:,1][:,np.newaxis] + theta[2]*X[:,2][:,np.newaxis] + theta[3]*X[:,3][:,np.newaxis]

        p = 1/(1+np.exp(-z))

        y = np.random.binomial(n=1, p=p, size=(ds_meta[c]["n_instances"],1)).ravel()

        if(ds_meta[c]["noise_added"] == "Yes"):
            noisy_rate = 0.2
            pos_idx = (y == 1).nonzero()[0]
            noisy_inst = int(noisy_rate * pos_idx.size)
            sel_pos_idx =   np.random.choice(pos_idx, noisy_inst,replace=False)  

            for j, m in enumerate(means):
  
                if (m == 0): # Normal distribution
                    
                    e = np.random.normal(size=sel_pos_idx.size)
                    X[sel_pos_idx, j] += e

                elif (m == 0.5): # Bernoulli distribution
                

                    X[sel_pos_idx, j] = np.where(X[sel_pos_idx, j]==0, 1, 0)         

        
        np.save(f'{ds_main_dir}{c}_X.npy', X)
        np.save(f'{ds_main_dir}{c}_y.npy', y)

        
        dataset_descriptions.loc[i] = [c, ds_meta[c]["n_instances"], ds_meta[c]["n_features"], ds_meta[c]["n_cat_features"], X.shape[1], ds_meta[c]["class_distribution"], ds_meta[c]["noise_added"]]
        i+=1

    dataset_descriptions = dataset_descriptions.round(4)
    dataset_descriptions.index += 1 
    dataset_descriptions.to_csv(f'{output_table_main_dir}dataset_descriptons.csv')

        
    print("Simulated data generation completed.")



def split_datasets (ds_meta, n_splits, train_ratio, validation_ratio, test_ratio, random_state):

    splitted_datasets = {}

    for c in ds_meta:

        X = np.load(f'{ds_main_dir}{c}_X.npy')
        y = np.load(f'{ds_main_dir}{c}_y.npy')
        
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=random_state)
        
        splitted_datasets[c] = []

        for (train_index, test_index) in sss.split(X, y):

            # Split into training and testing data
            X_train, X_test, y_train, y_test = X[train_index].copy(), X[test_index].copy(), y[train_index].copy(), y[test_index].copy()

            X_validation, y_validation = None, None # validation data to use for calibration, ROCCH method, etc.
            if (validation_ratio):
                # Split training data into training and calibration data
                X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, stratify=y_train, test_size=(validation_ratio/(validation_ratio+train_ratio)), random_state=random_state)

            
            splitted_datasets[c].append((X_train, X_test, X_validation, y_train, y_test, y_validation))

    
    return splitted_datasets

