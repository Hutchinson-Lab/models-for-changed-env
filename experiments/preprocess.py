'''

'''


import os, io, zipfile, requests, tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler



ds_dir = './experiments/datasets/'
output_table_dir = './experiments/tables/'

def download_datasets (ds_meta):
    
    print("\nDownloading datasets:")
    
    if not os.path.isdir(ds_dir):
        os.makedirs(ds_dir)
   
    for c in (pbar := tqdm.tqdm(ds_meta.keys())):
        pbar.set_description(f'Downloading "{c}"')
        r = requests.get(ds_meta[c]["url"])
        file_content = r.content

        if ds_meta[c]["url"].endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(r.content), 'r')
            file_content = z.open(ds_meta[c]["local-path"]).read()

        with open(f'{ds_dir}{ds_meta[c]["filename"]}', 'wb') as f:
            f.write(file_content)
        
    print("Downloads completed.")
        # print(c)

def run_preprocessing (ds_meta, K, train_ratio, separated_ratio, test_ratio, random_state):
    '''
    
    
    '''

    print("Preprocessing datasets:")
    
    preprocessed_datasets = {}

    dataset_descriptions = pd.DataFrame(columns=('Data Set','Instances','Non-missing Instances', 'Features', 'Categorical Features', 'Class Balance'))
    i = 0
    
    for c in (pbar := tqdm.tqdm(ds_meta.keys())):

        pbar.set_description(f'Preprocessing "{c}"')

        data = None
        if (ds_meta[c]["filetype"]==".xlsx"):
            data = pd.read_excel(f'{ds_dir}{ds_meta[c]["filename"]}', 
                                na_values=ds_meta[c]["missing-data-identifier"])
            
        elif (ds_meta[c]["filetype"]==".dat"):
            data = pd.read_csv( f'{ds_dir}{ds_meta[c]["filename"]}', 
                                na_values=ds_meta[c]["missing-data-identifier"],
                                skipinitialspace=True,
                                delim_whitespace=True)
        else:
            data = pd.read_csv( f'{ds_dir}{ds_meta[c]["filename"]}', 
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

        dataset_descriptions.loc[i] = [c, instances_num, nonmissing_instances_num, features_num, categorical_feature_num, class_distr]
        i+=1

        sss = StratifiedShuffleSplit(n_splits=K, test_size=test_ratio, random_state=random_state)
        X = data_x.values
        y = data_y.values.astype('int')
        
        preprocessed_datasets[c] = []
        for j, (train_index, test_index) in enumerate(sss.split(X, y)):
            # Split into training and testing data
            X_train, X_test, y_train, y_test = X[train_index].copy(), X[test_index].copy(), y[train_index].copy(), y[test_index].copy()

            X_separated, y_separated = None, None # Separated data to use for calibration, ROCCH method, etc.
            if (separated_ratio):
                # Split training data into training and calibration data
                X_train, X_separated, y_train, y_separated = train_test_split(X_train, y_train, stratify=y_train, test_size=(separated_ratio/(separated_ratio+train_ratio)), random_state=random_state)

            # Scale features
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            if (separated_ratio):
                X_separated = scaler.transform(X_separated)
            
            preprocessed_datasets[c].append((X_train, X_test, X_separated, y_train, y_test, y_separated))

    
    if not os.path.exists(output_table_dir):
        os.makedirs(output_table_dir)

    dataset_descriptions = dataset_descriptions.round(4)
    dataset_descriptions.index += 1 
    dataset_descriptions.to_csv(f'{output_table_dir}dataset_descriptons.csv')

    print("Preprocessing completed.")
    return preprocessed_datasets
    # print(dataset_descriptions)

