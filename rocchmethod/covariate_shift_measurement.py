import numpy as np
from scipy.stats import wasserstein_distance, energy_distance, cramervonmises_2samp
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from .mmd import mmd_linear


def average_wasserstein_distance(X_1, X_2):
    
    n_features = X_1.shape[1]

    avg_w_dist = 0
    for i in range(n_features):
        avg_w_dist += wasserstein_distance(X_1[:,i], X_2[:,i])
    
    avg_w_dist /= n_features

    return avg_w_dist 

def average_energy_distance(X_1, X_2):
    
    n_features = X_1.shape[1]

    avg_w_dist = 0
    for i in range(n_features):
        avg_w_dist += energy_distance(X_1[:,i], X_2[:,i])
    
    avg_w_dist /= n_features

    return avg_w_dist


def average_auc_phi (X_1, X_2, repeats=3, test_ratio=0.2, random_state=0):
    
    min_samples = min(X_1.shape[0], X_2.shape[0])
    X_1, X_2 = X_1[:min_samples, :], X_2[:min_samples, :]
    X_1 = np.hstack((X_1, np.zeros((min_samples,1), dtype=np.int8)))
    X_2 = np.hstack((X_2, np.ones((min_samples,1), dtype=np.int8)))
    data = np.vstack((X_1,X_2))
    X = data[:, :-1]
    y = data[:, -1]

    auc_avg = 0
    phi_avg = 0

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=test_ratio, random_state=random_state)
        
    for (train_index, test_index) in sss.split(X, y):
        # Split into training and testing data
        X_train, X_test, y_train, y_test = X[train_index].copy(), X[test_index].copy(), y[train_index].copy(), y[test_index].copy()
        
        # Scale features
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=0, n_jobs=-1)
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:,1]
        y_hat = model.predict(X_test)

        phi_avg += matthews_corrcoef(y_test, y_hat)
        auc_avg += roc_auc_score(y_test, y_score)
        
    auc_avg /= repeats
    phi_avg /= repeats

    return auc_avg, phi_avg


def average_cramervonmises (X_1, X_2):
    n_features = X_1.shape[1]

    avg_cramervonmises_criterion = 0
    for i in range(n_features):
        res = cramervonmises_2samp(X_1[:,i], X_2[:,i])
        avg_cramervonmises_criterion += res.statistic
    
    avg_cramervonmises_criterion /= n_features

    return avg_cramervonmises_criterion 
