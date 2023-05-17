'''
Experiment: Evaluate practicality and applicability of the ROC Convex Hull Method.

'''

import tqdm
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .preprocess import split_datasets


import sys, os
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..' )))


from rocchmethod.covariate_shift_measurement import average_wasserstein_distance, average_energy_distance, average_auc_phi, average_cramervonmises, mmd_linear
from rocchmethod.class_utils import normalized_cost, expected_cost, unique_cls_distr, impose_class_distr
from rocchmethod.rocchmethod import rocch_method, classifiers_on_rocch

# from plot_descriptions import selected_causal_graphs

# No of repeats/splits
repeats_range = [30] 

# split_ratios[i] = [train_ratio, validation_ratio, test_ratio]
split_ratio_range = [ 
    [0.2, 0.6, 0.2], # train=20%, validation=60%, test=20%
    [0.4, 0.4, 0.2], # train=40%, validation=40%, test=20%
    [0.6, 0.2, 0.2], # train=60%, validation=20%, test=20%
    [0.4, 0.2, 0.4], # train=40%, validation=20%, test=40%
    [0.2, 0.2, 0.6], # train=20%, validation=20%, test=60%
    [0.2, 0.4, 0.4], # train=20%, validation=40%, test=40%
]


# environment = ["%/100 of original clas distr.", "FP cost", "FN cost"]
# "%/100 of original clas distr." represents the scaled percentage of the original class distribution (of training set) in the imprecise environment/test set
# A value of 1.0 indicates that the test set has the same class distriution as the original/training set.
# A value of 0.5 indicates that the test set is equal to the original class distribution multiplied by 0.5 (divided by 2).
# A value of 1.25 indicates that the test set is equal to the original class distribution mulitplied by 1.25 (divided by 4/5)  

# Set of class and cost distributions (imprecise environments) we will be testing on,
environments = [

    [1.0, 1.0, 1.0], # control, same class distribution as training data, uniform cost distribution
    [1.0, 1.0, 1.5], # same class distribution as training data, FN cost is 1.5 times FP cost
    [1.0, 1.0, 3.0], # same class distribution as training data, FN cost is 3 times FP cost

    [0.5, 1.0, 1.0], # halved class distribution, uniform cost distribution
    [0.5, 1.0, 1.5], # halved class distribution,  FN cost is 1.5 times FP cost
    [0.5, 1.0, 3.0], # halved class distribution,  FN cost is 3 times FP cost
   
    [0.75, 1.0, 1.0], # 3/4 times class distribution, uniform cost distribution 
    [0.75, 1.0, 1.5], # 3/4 times class distribution,  FN cost is 1.5 times FP cost
    [0.75, 1.0, 3.0], # 3/4 times class distribution,  FN cost is 3 times FP cost
    
    [1.25, 1.0, 1.0], # 5/4 times class distribution, uniform cost distribution
    [1.25, 1.0, 1.5], # 5/4 times class distribution,  FN cost is 1.5 times FP cost
    [1.25, 1.0, 3.0], # 5/4 times class distribution,  FN cost is 3 times FP cost
    
]

oversampling_undersampling_methods = [
    ['Random', 'Random'],
    ['SMOTE', 'NearMiss'],
]

random_state = 0




# ----------------------------------------------------------------------------------------
# For temporary experimentation only
# ----------------------------------------------------------------------------------------

# repeats_range = [3]

# split_ratio_range = [ 
#     [0.4, 0.4, 0.2], # train=40%, validation=20%, test=40%
# ]
# environments = [

#     [0.5, 1.0, 1.0], # halved class distribution, uniform cost distribution
#     [1.25, 1.0, 1.0], # halved class distribution, uniform cost distribution
   
# ]
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------



environments_cls_distr = unique_cls_distr(environments)


output_table_dir = './experiments/Synthetic/tables/'





def run_experiments(ds_meta):
    '''
    '''
    dataset_descriptions = pd.read_csv(f'{output_table_dir}dataset_descriptons.csv') # Saved during preprocessing
    original_cls_distr = dict(zip(dataset_descriptions['Data Set'], dataset_descriptions['Class Balance']))
    ds_keys = list(dataset_descriptions['Data Set'])
    
    performance_df = pd.DataFrame(columns=(
        'Repeats',
        'Train Ratio',
        'Validation Ratio',
        'Test Ratio',
        'Data Set',
        'Test Size',
        'Train Class Distr.',
        'Test to Train Class Distr. Ratio',
        'Test Class Distr.',
        'FP cost',
        'FN cost',
        'Oversampling Method',
        'Undersampling Method',
        'Avg. Wasserstein Dist.',
        'Avg. Energy Dist.',
        'MMD',
        'Avg. AUC (COVSHIFT)',
        'Avg. Phi',
        'Avg. Cramer-von Mises Criterion',
        'No. Test Instances',
        'Split No.',
        'Optimal FPR (ROCCH Method)',
        'Optimal TPR (ROCCH Method)',
        'Isoperformance_df Line Slope',
        'Isoperformance_df Line Intercept',
        'Classifier',
        'Threshold',
        'FPR',
        'TPR',
        'Normalized Cost',
        'Expected Cost',
        'Accuracy',
        'F1-score',
        'Accuracy (Validation)',
        'F1-score (Validation)',
        'Normalized Cost (Validation)',
        'Expected Cost (Validation)',
        )
    )
    c = 0

    


    for repeats in repeats_range:

        for split_ratio in split_ratio_range:

            train_ratio, validation_ratio, test_ratio = split_ratio[0], split_ratio[1], split_ratio[2] 

            print(f'\nRepeats: {repeats}, Train Ratio: {train_ratio}, Validation Ratio: {validation_ratio}, Test Ratio: {test_ratio}')    
            
            preprocessed_ds = split_datasets(ds_meta, repeats, train_ratio, validation_ratio, test_ratio, random_state=random_state)
            
            for ds_key in (pbar := tqdm.tqdm(preprocessed_ds.keys())):
                
                pbar.set_description(f'Running experiments on "{ds_key}"')


                for split_num in range(repeats):

                    X_train, X_test, X_validation, y_train, y_test, y_validation = preprocessed_ds[ds_key][split_num]
            
                    # Train classifiers and predict on (existing) test set
                    models = { 
                        # Default parameters used for models from sklearn 1.2.0
                        # same random state across splits, but different training and test data across splits
                        'A' : RandomForestClassifier(random_state=0),
                        'B' : LogisticRegression(random_state=0),
                        'C' : KNeighborsClassifier(),
                    }

                    y_hats = []

                    for m in models:
                        
                        models[m].fit(X_train, y_train)
                        
                        y_hats.append(models[m].predict_proba(X_validation)[:,1])
                        
                        
                    fpr_list, tpr_list, threshold_list, rocch_fpr, rocch_tpr, optimals = rocch_method(
                                                                                                y_validation,
                                                                                                y_hats,
                                                                                                environments
                                                                                            )
                    rocch_classifiers, rocch_thresholds = classifiers_on_rocch(fpr_list,tpr_list, threshold_list, rocch_fpr, rocch_tpr)
                    



                    for os_us in oversampling_undersampling_methods:

                        test_cls_distr = {cls_distr : impose_class_distr(X_test, y_test, (original_cls_distr[ds_key] * cls_distr), os_us[0], os_us[1], random_state) for cls_distr in environments_cls_distr}


                        for i, environment in enumerate(environments):

                            X_test_env, y_test_env = test_cls_distr[environment[0]]

                            avg_w_dist = average_wasserstein_distance(X_train, X_test_env)
                            avg_e_dist = average_energy_distance(X_train, X_test_env)
                            

                            mmd = mmd_linear(X_train, X_test_env)
                            avg_auc, avg_phi = average_auc_phi(X_train, X_test_env)
                            
                            
                            avg_cvmc = average_cramervonmises(X_train, X_test_env)

                            
                            for j, rocch_fpr_t in enumerate(rocch_fpr):
                                
                                for k, clfs in enumerate(rocch_classifiers[j]):
                                    predictions = models[list(models.keys())[k]].predict_proba(X_test_env)[:,1]
                                    predictions_hard = np.where(1, predictions>=rocch_thresholds[j][k], 0)

                                    predictions_sep = models[list(models.keys())[k]].predict_proba(X_validation)[:,1]
                                    predictions_sep_hard = np.where(1, predictions_sep>=rocch_thresholds[j][k], 0)

                                    norm_cost = normalized_cost(y_test_env, predictions_hard, environment[1], environment[2])
                                    exp_cost = expected_cost(y_test_env, predictions_hard, environment[1], environment[2])
                                    acc = accuracy_score(y_test_env, predictions_hard)
                                    f1_s = f1_score(y_test_env, predictions_hard)

                                    acc_sep = accuracy_score(y_validation, predictions_sep_hard)
                                    f1_s_sep = f1_score(y_validation, predictions_sep_hard)
                                    norm_cost_sep = normalized_cost(y_validation, predictions_sep_hard, environment[1], environment[2])
                                    exp_cost_sep = expected_cost(y_validation, predictions_sep_hard, environment[1], environment[2])

                                    performance_df.loc[c] = [
                                        repeats,
                                        train_ratio,
                                        validation_ratio,
                                        test_ratio,
                                        ds_key,
                                        predictions_hard.shape[0],
                                        original_cls_distr[ds_key],
                                        environment[0],
                                        original_cls_distr[ds_key] * environment[0],
                                        environment[1], 
                                        environment[2],
                                        os_us[0],
                                        os_us[1],
                                        avg_w_dist,
                                        avg_e_dist,
                                        mmd,
                                        avg_auc,
                                        avg_phi,
                                        avg_cvmc,
                                        y_test_env.shape[0],
                                        split_num,
                                        optimals[i][0],
                                        optimals[i][1],
                                        optimals[i][2],
                                        optimals[i][3],
                                        list(models.keys())[clfs],
                                        rocch_thresholds[j][k],
                                        rocch_fpr_t,
                                        rocch_tpr[j],
                                        norm_cost,
                                        exp_cost,
                                        acc,
                                        f1_s,
                                        acc_sep,
                                        f1_s_sep,
                                        norm_cost_sep,
                                        exp_cost_sep,
                                        ]
                                    c += 1
        
        
    performance_df = performance_df.round(4)
    performance_df.to_csv(f'{output_table_dir}performance.csv')

    # performance_df = pd.read_csv(f'{output_table_dir}performance.csv')

    
    performance_summarized_df = pd.DataFrame(columns=(
        'Repeats',
        'Train Ratio',
        'Validation Ratio',
        'Test Ratio',
        'Data Set',
        'Train Class Distr.',
        'Test to Train Class Distr. Ratio',
        'Test Class Distr.',
        'FP cost',
        'FN cost',
        'Oversampling Method',
        'Undersampling Method',
        'Avg. Wasserstein Dist.',
        'Avg. Energy Dist.',
        'MMD',
        'Avg. AUC (COVSHIFT)',
        'Avg. Phi',
        'Avg. Cramer-von Mises Criterion',
        'Split No.',
        'Optimal FPR (ROCCH Method)',
        'Optimal TPR (ROCCH Method)',
        'Optimal Point Normalized Cost (ROCCH Method)',
        'Optimal Point Expected Cost (ROCCH Method)',
        'Optimal FPR (Norm-Cost-Min)',
        'Optimal TPR (Norm-Cost-Min)',
        'Optimal Point Normalized Cost (Norm-Cost-Min)',
        'Optimal Point Expected Cost (Norm-Cost-Min)',
        'Optimal FPR (Exp-Cost-Min)',
        'Optimal TPR (Exp-Cost-Min)',
        'Optimal Point Normalized Cost (Exp-Cost-Min)',
        'Optimal Point Expected Cost (Exp-Cost-Min)',
        'Optimal FPR (Accuracy-Max)',
        'Optimal TPR (Accuracy-Max)',
        'Optimal Point Normalized Cost (Accuracy-Max)',
        'Optimal Point Expected Cost (Accuracy-Max)',
        'Optimal FPR (F1-score-Max)',
        'Optimal TPR (F1-score-Max)',
        'Optimal Point Normalized Cost (F1-score-Max)',
        'Optimal Point Expected Cost (F1-score-Max)',
        'Optimal FPR (Oracle-Norm)',
        'Optimal TPR (Oracle-Norm)',
        'Optimal Point Normalized Cost (Oracle-Norm)',
        'Optimal Point Expected Cost (Oracle-Norm)',
        'Optimal FPR (Oracle-Exp)',
        'Optimal TPR (Oracle-Exp)',
        'Optimal Point Normalized Cost (Oracle-Exp)',
        'Optimal Point Expected Cost (Oracle-Exp)',
        'Distance between ROCCHM and Oracle-Norm',
        'Distance between Norm-Cost-Min and Oracle-Norm',
        'Distance between Exp-Cost-Min and Oracle-Norm',
        'Distance between Accuracy-Max and Oracle-Norm',
        'Distance between F1-score-Max and Oracle-Norm',
        'Distance between ROCCHM and Oracle-Exp',
        'Distance between Norm-Cost-Min and Oracle-Exp',
        'Distance between Exp-Cost-Min and Oracle-Exp',
        'Distance between Accuracy-Max and Oracle-Exp',
        'Distance between F1-score-Max and Oracle-Exp',
        
        
        )
    )

    c = 0

    print("\nSummarizing and saving results.")
    for repeats in repeats_range:
        for split_ratio in split_ratio_range:

            train_ratio, validation_ratio, test_ratio = split_ratio[0], split_ratio[1], split_ratio[2] 
            
            for ds_key in ds_keys:
                for split_num in range(repeats):

                    for os_us in oversampling_undersampling_methods:
                        for i, environment in enumerate(environments):   
                            current_slice_idx = (
                                            (performance_df['Repeats'] == repeats) & 
                                            (performance_df['Train Ratio'] == train_ratio) &
                                            (performance_df['Validation Ratio'] == validation_ratio) &
                                            (performance_df['Test Ratio'] == test_ratio) &
                                            (performance_df['Data Set'] == ds_key) &
                                            (performance_df['Split No.'] == split_num) &
                                            (performance_df['Test to Train Class Distr. Ratio'] == environment[0]) &
                                            (performance_df['FP cost'] == environment[1]) &
                                            (performance_df['FN cost'] == environment[2]) &
                                            (performance_df['Oversampling Method'] == os_us[0]) &
                                            (performance_df['Undersampling Method'] == os_us[1])

                                        ) 

                            current_df = performance_df.loc[current_slice_idx]


                            # Cost of Optimal Point selected by ROCCH Method
                            current_optimal_df = current_df[
                                            (current_df['Optimal FPR (ROCCH Method)'] == current_df['FPR']) & 
                                            (current_df['Optimal TPR (ROCCH Method)'] == current_df['TPR'])
                                        ]
                            
                            # We might get multiple discrete classifiers, if they come from different continuous classifeirs, but with same FPR and TPR values
                            # We take the discrete classifier with the lowest corresponding cost
                            rocchm_optimal_point_norm_cost = current_optimal_df['Normalized Cost'].min()
                            rocchm_optimal_point_exp_cost = current_optimal_df['Expected Cost'].min()
                            

                            # Normalized cost minimizing FPR and TPR
                            normcostmin_idx = current_df['Normalized Cost (Validation)'].idxmin()
                            normcostmin_optimal = [current_df['FPR'].loc[normcostmin_idx], current_df['TPR'].loc[normcostmin_idx]]

                            # Cost of normalized cost minimizing FPR and TPR
                            normcostmin_optimal_point_norm_cost = current_df['Normalized Cost'].loc[normcostmin_idx]
                            normcostmin_optimal_point_exp_cost = current_df['Expected Cost'].loc[normcostmin_idx]


                            # Expected cost minimizing FPR and TPR
                            expcostmin_idx = current_df['Expected Cost (Validation)'].idxmin()
                            expcostmin_optimal = [current_df['FPR'].loc[expcostmin_idx], current_df['TPR'].loc[expcostmin_idx]]

                            # Cost of expected cost minimizing FPR and TPR
                            expcostmin_optimal_point_norm_cost = current_df['Normalized Cost'].loc[expcostmin_idx]
                            expcostmin_optimal_point_exp_cost = current_df['Expected Cost'].loc[expcostmin_idx]

                            # Accuracy maximizing FPR and TPR
                            accumax_idx = current_df['Accuracy (Validation)'].idxmax()
                            accumax_optimal = [current_df['FPR'].loc[accumax_idx], current_df['TPR'].loc[accumax_idx]]

                            # Cost of accuracy maximizing FPR and TPR
                            accumax_optimal_point_norm_cost = current_df['Normalized Cost'].loc[accumax_idx]
                            accumax_optimal_point_exp_cost = current_df['Expected Cost'].loc[accumax_idx]

                            # F1-score maximizing FPR and TPR
                            fonemax_idx = current_df['F1-score (Validation)'].idxmax()
                            fonemax_optimal = [current_df['FPR'].loc[fonemax_idx], current_df['TPR'].loc[fonemax_idx]]

                            # Cost of F1-score maximizing FPR and TPR
                            fonemax_optimal_point_norm_cost = current_df['Normalized Cost'].loc[fonemax_idx]
                            fonemax_optimal_point_exp_cost = current_df['Expected Cost'].loc[fonemax_idx]

                            # Oracle FPR and TPR based on Normalized Cost
                            oracle_min_norm_cost_idx = current_df['Normalized Cost'].idxmin()
                            oracle_norm_optimal = [current_df['FPR'].loc[oracle_min_norm_cost_idx], current_df['TPR'].loc[oracle_min_norm_cost_idx]]
                            
                            # Cost of Oracle Optimal Point based on Normalized Cost
                            oracle_norm_optimal_point_norm_cost = current_df['Normalized Cost'].loc[oracle_min_norm_cost_idx]
                            oracle_norm_optimal_point_exp_cost = current_df['Expected Cost'].loc[oracle_min_norm_cost_idx]

                            # Oracle FPR and TPR based on Expected Cost
                            oracle_min_exp_cost_idx = current_df['Expected Cost'].idxmin()
                            oracle_exp_optimal = [current_df['FPR'].loc[oracle_min_exp_cost_idx], current_df['TPR'].loc[oracle_min_exp_cost_idx]]
                            
                            # Cost of Oracle Optimal Point based on Expected Cost
                            oracle_exp_optimal_point_norm_cost = current_df['Normalized Cost'].loc[oracle_min_exp_cost_idx]
                            oracle_exp_optimal_point_exp_cost = current_df['Expected Cost'].loc[oracle_min_exp_cost_idx]



                            # Distance (in ROC space) between Optimal Point selected by ROCCH Method and Oracle Optimal Point based on Normalized Cost
                            rocchm_optimal = [current_df['Optimal FPR (ROCCH Method)'].iloc[0], current_df['Optimal TPR (ROCCH Method)'].iloc[0]] 
                            distance_rocchm_oracle_norm = np.linalg.norm (np.array(rocchm_optimal)-np.array(oracle_norm_optimal))

                            # Distance (in ROC space) between Optimal Point based on Norm-Cost-Min and Oracle Optimal Point based on Normalized Cost
                            distance_normcostmin_oracle_norm = np.linalg.norm (np.array(normcostmin_optimal)-np.array(oracle_norm_optimal))

                            # Distance (in ROC space) between Optimal Point based on Exp-Cost-Min and Oracle Optimal Point based on Normalized Cost
                            distance_expcostmin_oracle_norm = np.linalg.norm (np.array(expcostmin_optimal)-np.array(oracle_norm_optimal))

                            # Distance (in ROC space) between Optimal Point based on Accuracy-Max and Oracle Optimal Point based on Normalized Cost
                            distance_accumax_oracle_norm = np.linalg.norm (np.array(accumax_optimal)-np.array(oracle_norm_optimal))

                            # Distance (in ROC space) between Optimal Point based on Accuracy-Max and Oracle Optimal Point based on Normalized Cost
                            distance_fonemax_oracle_norm = np.linalg.norm (np.array(fonemax_optimal)-np.array(oracle_norm_optimal))


                            # Distance (in ROC space) between Optimal Point selected by ROCCH Method and Oracle Optimal Point based on Expected Cost
                            rocchm_optimal = [current_df['Optimal FPR (ROCCH Method)'].iloc[0], current_df['Optimal TPR (ROCCH Method)'].iloc[0]] 
                            distance_rocchm_oracle_exp = np.linalg.norm (np.array(rocchm_optimal)-np.array(oracle_norm_optimal))

                            # Distance (in ROC space) between Optimal Point based on Norm-Cost-Min and Oracle Optimal Point based on Expected Cost
                            distance_normcostmin_oracle_exp = np.linalg.norm (np.array(normcostmin_optimal)-np.array(oracle_exp_optimal))

                            # Distance (in ROC space) between Optimal Point based on Exp-Cost-Min and Oracle Optimal Point based on Expected Cost
                            distance_expcostmin_oracle_exp = np.linalg.norm (np.array(expcostmin_optimal)-np.array(oracle_exp_optimal))

                            # Distance (in ROC space) between Optimal Point based on Accuracy-Max and Oracle Optimal Point based on Expected Cost
                            distance_accumax_oracle_exp = np.linalg.norm (np.array(accumax_optimal)-np.array(oracle_exp_optimal))

                            # Distance (in ROC space) between Optimal Point based on Accuracy-Max and Oracle Optimal Point based on Expected Cost
                            distance_fonemax_oracle_exp = np.linalg.norm (np.array(fonemax_optimal)-np.array(oracle_exp_optimal))


                            # # Test set size, for calculating avg. exp. cost
                            # test_size = current_df['Test Size'].iloc[0]
                            

                            performance_summarized_df.loc[c] = [
                                        repeats,
                                        current_df['Train Ratio'].iloc[0],
                                        current_df['Validation Ratio'].iloc[0],
                                        current_df['Test Ratio'].iloc[0],
                                        ds_key,
                                        original_cls_distr[ds_key],
                                        environment[0],
                                        original_cls_distr[ds_key] * environment[0],
                                        environment[1], 
                                        environment[2],
                                        os_us[0],
                                        os_us[1],
                                        current_df['Avg. Wasserstein Dist.'].iloc[0],
                                        current_df['Avg. Energy Dist.'].iloc[0],
                                        current_df['MMD'].iloc[0],
                                        current_df['Avg. AUC (COVSHIFT)'].iloc[0],
                                        current_df['Avg. Phi'].iloc[0],
                                        current_df['Avg. Cramer-von Mises Criterion'].iloc[0],
                                        split_num,
                                        current_df['Optimal FPR (ROCCH Method)'].iloc[0],
                                        current_df['Optimal TPR (ROCCH Method)'].iloc[0],
                                        rocchm_optimal_point_norm_cost,
                                        rocchm_optimal_point_exp_cost,
                                        normcostmin_optimal[0],
                                        normcostmin_optimal[1],
                                        normcostmin_optimal_point_norm_cost,
                                        normcostmin_optimal_point_exp_cost,
                                        expcostmin_optimal[0],
                                        expcostmin_optimal[1],
                                        expcostmin_optimal_point_norm_cost,
                                        expcostmin_optimal_point_exp_cost,
                                        accumax_optimal[0],
                                        accumax_optimal[1],
                                        accumax_optimal_point_norm_cost,
                                        accumax_optimal_point_exp_cost,
                                        fonemax_optimal[0],
                                        fonemax_optimal[1],
                                        fonemax_optimal_point_norm_cost,
                                        fonemax_optimal_point_exp_cost,
                                        oracle_norm_optimal[0],
                                        oracle_norm_optimal[1],
                                        oracle_norm_optimal_point_norm_cost,
                                        oracle_norm_optimal_point_exp_cost,
                                        oracle_exp_optimal[0],
                                        oracle_exp_optimal[1],
                                        oracle_exp_optimal_point_norm_cost,
                                        oracle_exp_optimal_point_exp_cost,
                                        distance_rocchm_oracle_norm,
                                        distance_normcostmin_oracle_norm,
                                        distance_expcostmin_oracle_norm,
                                        distance_accumax_oracle_norm,
                                        distance_fonemax_oracle_norm,
                                        distance_rocchm_oracle_exp,
                                        distance_normcostmin_oracle_exp,
                                        distance_expcostmin_oracle_exp,
                                        distance_accumax_oracle_exp,
                                        distance_fonemax_oracle_exp,

                                        
                                    ]
                            c += 1



    performance_summarized_df = performance_summarized_df.round(4)
    performance_summarized_df.to_csv(f'{output_table_dir}performance_summarized.csv')


    print("Experiments on Synthetic Data completed.")



