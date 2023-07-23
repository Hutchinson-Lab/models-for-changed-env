# Implementation of ROC Convex Hull Method

# Nahian Ahmed
# July 23, 2023

import numpy as np
from sklearn.metrics import roc_curve

from .rocch import rocch


def rocch_method (labels, predictions, environments):
    """
    @author: Nahian Ahmed
    
    ROC Convex Hull (ROCCH) Method (Provost and Fawcett, 1997, 1998, 2001)
    
        Foster Provost and Tom Fawcett. 
            Analysis and visualization of classifier performance with nonuniform class and cost distributions. 
            In Proceedings of AAAI-97 Workshop on AI Approaches to Fraud Detection & Risk Management, pages 57–63, 1997. 
            https://www.aaai.org/Papers/Workshops/1997/WS-97-07/WS97-07-011.pdf

        Foster Provost and Tom Fawcett. 
            Robust classification systems for imprecise environments.
            In AAAI/IAAI, pages 706–713, 1998. 
            https://www.aaai.org/Papers/AAAI/1998/AAAI98-100.pdf

        Foster Provost and Tom Fawcett. 
            Robust classification for imprecise environments. 
            Machine learning, 42(3):203–231, 2001. 
            https://link.springer.com/content/pdf/10.1023/A:1007601015854.pdf

        Code for computing the ROC convex hull was forked from Dr. Fayyaz Minhas' implementation (https://github.com/foxtrotmike/rocch) 
        
        Input:
            predictions: List of predictions on test set by classifiers. 
                len(predictions) = number of classifiers.
                len(predictions[i]) = number of instances in test set.
            environments: List of environments for which to find optimal discrete classifiers for.
                len(environments) = number of environments.

        Return:
            fpr_list:
            tpr_list:
            threshold_list:
            optimals: list of optimal points with each element having the format (optimal_fpr, optimal_tpr, slope, intercept)

    """
  

    # STEP 1: Construct ROC Curves for Continuous Classifiers (and Points in ROC Space for Discrete Classifiers)
    # Obtain ROC curves
    fpr_list = []
    tpr_list = []
    threshold_list = []
    for prediction in predictions:
        fpr, tpr, thresholds = roc_curve(labels, prediction, drop_intermediate=True)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        threshold_list.append(thresholds)

        
    # STEP 2: Find Convex Hull of all Classifiers
    fpr_all = np.concatenate(fpr_list)
    tpr_all = np.concatenate(tpr_list)
    rocch_fpr, rocch_tpr, _ = rocch(fpr_all, tpr_all)
    rocch_fpr = np.array(rocch_fpr)
    rocch_tpr = np.array(rocch_tpr)
    
    # STEP 3: Find Slopes of Iso-Performance Lines for Each Set of Class and Cost Distribution
    slopes = []
    for environment in environments:
        slopes.append(((1 - environment[0]) / environment[0]) * (environment[1] / environment[2]))


    # STEP 4: Find Optimal Classifiers as Discrete Points on ROC Convex Hull that Intersect Iso-Performance Lines
    
    optimals = []

    for slope in slopes:
        
        intercepts = []
        for i, rocch_fpr_c in enumerate(rocch_fpr):
            intercept = rocch_tpr[i] - (slope * rocch_fpr_c) # y = mx + b, b = y - mx 
            intercepts.append(intercept) 
        
        intersection_idx = np.argmax(intercepts)
            
        optimals.append((rocch_fpr[intersection_idx], rocch_tpr[intersection_idx], slope, intercepts[intersection_idx]))
    

    return fpr_list, tpr_list, threshold_list, rocch_fpr, rocch_tpr, optimals


def classifiers_on_rocch(fpr_list, tpr_list, threshold_list, rocch_fpr, rocch_tpr):
    n_classifiers = len(fpr_list)
    rocch_t = np.stack((rocch_fpr,rocch_tpr),axis=1)
    rocch_classifiers = []
    rocch_thresholds = []
    for i, rocch_t_ele in enumerate(rocch_t):
        rocch_classifiers.append([])
        rocch_thresholds.append([])
        
        for j in range(n_classifiers):
            roc_t = np.stack((fpr_list[j],tpr_list[j]),axis=1)
            idx = None 
            for k, roc_t_i in enumerate(roc_t):
                if ((roc_t_i == rocch_t_ele).all()):
                    idx = k
                    break
            if idx is not None:
                rocch_classifiers[i].append(j)
                rocch_thresholds[i].append(threshold_list[j][idx])    
        
    return rocch_classifiers, rocch_thresholds
            