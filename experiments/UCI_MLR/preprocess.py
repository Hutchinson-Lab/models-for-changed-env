# Preprocess UCI datasets

# Nahian Ahmed
# July 23, 2023

import os, io, zipfile, requests, tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))


ds_main_dir = './experiments/UCI_MLR/datasets/'
ds_raw_dir = './experiments/UCI_MLR/datasets/raw/'
ds_preprocessed_dir = './experiments/UCI_MLR/datasets/preprocessed/'
output_table_main_dir = './experiments/UCI_MLR/tables/'
output_plot_main_dir = './experiments/UCI_MLR/plots/'


def download_datasets (ds_meta):
    
    print("\nDownloading datasets:")
    
    if not os.path.isdir(ds_main_dir):
        os.makedirs(ds_main_dir)

    if not os.path.isdir(ds_raw_dir):
        os.makedirs(ds_raw_dir)
   
    for c in (pbar := tqdm.tqdm(ds_meta.keys())):
        pbar.set_description(f'Downloading "{c}"')
        r = requests.get(ds_meta[c]["url"])
        file_content = r.content

        if ds_meta[c]["url"].endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(r.content), 'r')
            file_content = z.open(ds_meta[c]["local-path"]).read()

        with open(f'{ds_raw_dir}{ds_meta[c]["filename"]}', 'wb') as f:
            f.write(file_content)
        
    print("Downloads completed.")
        # print(c)


def preprocess_datasets (ds_meta):
    
    print("Preprocessing datasets:")
    
    if not os.path.exists(ds_preprocessed_dir):
        os.makedirs(ds_preprocessed_dir)
    
    if not os.path.exists(output_table_main_dir):
        os.makedirs(output_table_main_dir)

    dataset_descriptions = pd.DataFrame(columns=('Dataset','Instances','Non-missing Instances', 'Features', 'Categorical Features', 'Features After One-Hot', 'Class Balance'))
    i = 0
    
    for c in (pbar := tqdm.tqdm(ds_meta.keys())):

        pbar.set_description(f'Preprocessing "{c}"')

        data = None
        if (ds_meta[c]["filetype"]==".xlsx"):
            data = pd.read_excel(f'{ds_raw_dir}{ds_meta[c]["filename"]}', 
                                na_values=ds_meta[c]["missing-data-identifier"])
            
        elif (ds_meta[c]["filetype"]==".dat"):
            data = pd.read_csv( f'{ds_raw_dir}{ds_meta[c]["filename"]}', 
                                na_values=ds_meta[c]["missing-data-identifier"],
                                skipinitialspace=True,
                                delim_whitespace=True)
        else:
            data = pd.read_csv( f'{ds_raw_dir}{ds_meta[c]["filename"]}', 
                                delimiter=ds_meta[c]["delimiter"],
                                na_values=ds_meta[c]["missing-data-identifier"],
                                skipinitialspace=True)
        
        instances_num = data.shape[0]
        data = data.dropna()
        nonmissing_instances_num = data.shape[0]
        features_num = data.shape[1] - 1
        categorical_feature_num = len(ds_meta[c]["categorical-feature-column-position"])

        data_x = data[data.columns[ds_meta[c]["feature-column-position"]]]
        
        data_x = pd.get_dummies(data_x, columns = data.columns[ds_meta[c]["categorical-feature-column-position"]])

        data_y = data[data.columns[ds_meta[c]["label-column-position"]]]
        
        data_y = data_y.replace(to_replace=ds_meta[c]["positive-label"], value=1).replace(to_replace=ds_meta[c]["negative-label"], value=0)
     
        class_distr = data_y.sum()/data_y.shape[0]

        
        X = data_x.values
        y = data_y.values.astype('int')

        # Save datasets
        np.save(f'{ds_preprocessed_dir}{c}_X.npy', X)
        np.save(f'{ds_preprocessed_dir}{c}_y.npy', y)

        # cls_dist_ratios = class_distance_ratio(X,y)
        
        dataset_descriptions.loc[i] = [c, instances_num, nonmissing_instances_num, features_num, categorical_feature_num, X.shape[1], class_distr]
        i+=1

    dataset_descriptions = dataset_descriptions.round(4)
    dataset_descriptions.index += 1 
    dataset_descriptions.to_csv(f'{output_table_main_dir}dataset_descriptons.csv')

    print("Preprocessing completed.")


def split_datasets (ds_meta, n_splits, train_ratio, validation_ratio, test_ratio, random_state):

    splitted_datasets = {}

    for c in ds_meta:

        X = np.load(f'{ds_preprocessed_dir}{c}_X.npy')
        y = np.load(f'{ds_preprocessed_dir}{c}_y.npy')
        
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=random_state)
        
        splitted_datasets[c] = []

        for (train_index, test_index) in sss.split(X, y):

            # Split into training and testing data
            X_train, X_test, y_train, y_test = X[train_index].copy(), X[test_index].copy(), y[train_index].copy(), y[test_index].copy()

            X_validation, y_validation = None, None # Validation data to use for calibration, ROCCH method, etc.
            if (validation_ratio):
                # Split training data into training and calibration data
                X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, stratify=y_train, test_size=(validation_ratio/(validation_ratio+train_ratio)), random_state=random_state)

            # Scale features
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            if (validation_ratio):
                X_validation = scaler.transform(X_validation)
            
            splitted_datasets[c].append((X_train, X_test, X_validation, y_train, y_test, y_validation))

    
    return splitted_datasets

