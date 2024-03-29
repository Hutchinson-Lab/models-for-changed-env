# Utilites for model selection under class and cost distribution change

# Nahian Ahmed
# July 23, 2023

import warnings
import numpy as np
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.metrics import confusion_matrix

def normalized_cost(y_true, y_pred, fpc, fnc):
    '''
    # Page 201 from Provost and Fawcett (Machine Learning, 2001)
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    fpr, tpr = None, None
    
    if (fp==0):
        fpr = 0.0
    else:
        fpr = fp/(fp+tn)

    if (tp==0):
        tpr = 0.0
    else:
        tpr = tp/(tp+fn)
        
    cost = (fpr * fpc) + ((1-tpr) * fnc) 
    return cost

def expected_cost(y_true, y_pred, fpc, fnc):
    '''
    # Page 208 from Provost and Fawcett (Machine Learning, 2001)
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    pos = (y_true == 1).sum()/y_true.size
    neg = 1 - pos

    fpr, tpr = None, None
    
    if (fp==0):
        fpr = 0.0
    else:
        fpr = fp/(fp+tn)

    if (tp==0):
        tpr = 0.0
    else:
        tpr = tp/(tp+fn)
        
    cost = (fpr * fpc * neg) + ((1-tpr) * fnc * pos) 
    return cost



# environment = ["%/100 of original clas distr.", "FP cost", "FN cost"]
# "%/100 of original clas distr." represents the scaled percentage of the original class distribution (of training set) in the changed environment/test set
# A value of 1.0 indicates that the test set has the same class distriution as the original/training set.
# A value of 0.5 indicates that the test set is equal to the original class distribution multiplied by 0.5 (divided by 2).
# A value of 1.25 indicates that the test set is equal to the original class distribution mulitplied by 1.25 (divided by 4/5)  


def unique_cls_distr(environments):
    
    unique_cls_distr = list(set([e[0] for e in environments]))

    return unique_cls_distr




def impose_class_distr(X, y, imposed_class_distr, oversampling, undersampling, random_state=0):
    """
    @author: Nahian Ahmed
    ROC Convex Hull (ROCCH) Method (Provost and Fawcett, 1997, 1998, 2001)
    
    
        Code for computing the ROC convex hull was forked from Dr. Fayyaz Minhas' implementation (https://github.com/foxtrotmike/rocch) 
        
        Input:
            predictions: List of predictions on test set by classifiers. 
                len(predictions) = number of classifiers.
                len(predictionos[0]) = number of instances in test set.
            environments: List of environments for which to find optimal discrete classifiers for.
                len(environments) = number of environments.

        Return:
            fpr_list:
            tpr_list:
            threshold_list:
            optimals: list of optimal points with each element having the format (optimal_fpr, optimal_tpr)
    """

    n_pos = y.sum()
    n_total = y.shape[0]
    original_class_distr = n_pos/n_total

    n_pos_imposed = int(n_total * imposed_class_distr)
    n_neg_imposed = n_total - n_pos_imposed

    # if negative, make zero
    n_pos_imposed = max(0, n_pos_imposed)
    n_neg_imposed = max(0, n_neg_imposed) 
    
    oversampling_strategy = None
    undersampling_strategy = None

    if (imposed_class_distr < original_class_distr): # Positives are to be undersampled, negatives are to be oversampled            
       oversampling_strategy = {0: n_neg_imposed}
       undersampling_strategy = {1: n_pos_imposed}

    elif (imposed_class_distr > original_class_distr): # Negatives are to be undersampled, positives are to be oversampled        
        oversampling_strategy = {1: n_pos_imposed}
        undersampling_strategy = {0: n_neg_imposed}
        
    else: # Imposed class distribution is equal to original class distribution 
        return X, y

    oversampling_method = None
    undersampling_method = None

    if oversampling == 'Random':
        oversampling_method = RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=random_state)
    elif oversampling == 'SMOTE':
        oversampling_method = SMOTE(sampling_strategy=oversampling_strategy, random_state=random_state, n_jobs=-1)

    if undersampling == 'Random':
        undersampling_method = RandomUnderSampler(sampling_strategy=oversampling_strategy, random_state=random_state)
    elif undersampling == 'NearMiss':
        undersampling_method = NearMiss(sampling_strategy=undersampling_strategy, n_jobs=-1)
    
    pipe = make_pipeline(
            oversampling_method,
            undersampling_method
        )

    # imblearn.over_sampling will throw user warning.
    # This is not an issue for us because we might oversample the majority class as well (and not just the minority class).
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore") 
        X, y = pipe.fit_resample(X, y)
    
    return X, y



