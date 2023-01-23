'''

'''
import warnings
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from scipy.spatial import ConvexHull
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss




# environment = ["%/100 of original clas distr.", "FP cost", "FN cost"]
# "%/100 of original clas distr." represents the scaled percentage of the original class distribution (of training set) in the imprecise environment/test set
# A value of 1.0 indicates that the test set has the same class distriution as the original/training set.
# A value of 0.5 indicates that the test set is equal to the original class distribution multiplied by 0.5 (divided by 2).
# A value of 1.25 indicates that the test set is equal to the original class distribution mulitplied by 1.25 (divided by 4/5)  


def unique_cls_distr(environments):
    
    unique_cls_distr = list(set([e[0] for e in environments]))

    return unique_cls_distr




def impose_class_distr(X, y, imposed_class_distr, random_state=1):
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
    
    smote_strategy = None
    nearmiss_strategy = None
    if (imposed_class_distr < original_class_distr): # Positives are to be undersampled, negatives are to be oversampled
                
       smote_strategy = {0: n_neg_imposed}
       nearmiss_strategy = {1: n_pos_imposed}

    elif (imposed_class_distr > original_class_distr): # Negatives are to be undersampled, positives are to be oversampled
        
        smote_strategy = {1: n_pos_imposed}
        nearmiss_strategy = {0: n_neg_imposed}
        
    else: # Imposed class distribution is equal to original class distribution
        
        return X, y

    pipe = make_pipeline(
            SMOTE(sampling_strategy=smote_strategy, random_state=random_state, n_jobs=-1),
            NearMiss(sampling_strategy=nearmiss_strategy, n_jobs=-1)
        )

    # imblearn.over_sampling will throw user warning.
    # This is not an issue for use because we might oversample the majority class as well (and not just the minority class).
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore") 
        X, y = pipe.fit_resample(X, y)
    
    return X, y



# Source: https://github.com/foxtrotmike/rocch
def rocch(fpr0,tpr0):
    """
    @author: Dr. Fayyaz Minhas (http://faculty.pieas.edu.pk/fayyaz/)
    Construct the convex hull of a Receiver Operating Characteristic (ROC) curve
        Input:
            fpr0: List of false positive rates in range [0,1]
            tpr0: List of true positive rates in range [0,1]
                fpr0,tpr0 can be obtained from sklearn.metrics.roc_curve or 
                    any other packages such as pyml
        Return:
            F: list of false positive rates on the convex hull
            T: list of true positive rates on the convex hull
                plt.plot(F,T) will plot the convex hull
            auc: Area under the ROC Convex hull
    """
    fpr = np.array([0]+list(fpr0)+[1.0,1,0])
    tpr = np.array([0]+list(tpr0)+[1.0,0,0])
    hull = ConvexHull(np.vstack((fpr,tpr)).T)
    vert = hull.vertices
    vert = vert[np.argsort(fpr[vert])]  
    F = [0]
    T = [0]
    for v in vert:
        ft = (fpr[v],tpr[v])
        if ft==(0,0) or ft==(1,1) or ft==(1,0):
            continue
        F+=[fpr[v]]
        T+=[tpr[v]]
    F+=[1]
    T+=[1]
    auc = np.trapz(T,F)
    return F,T,auc




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
            

def expected_cost(y_true, y_pred, fpc, fnc):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = (fp * fpc) + (fn * fnc) # Pagee 44 from Provost and Fawcett (KDD 1997)
    return cost

# Need to implement metrics to find distance between points.
