# import numpy as np
# from imblearn.over_sampling import SMOTE, ADASYN

# random_state = 1

# a = np.array(

#     [
#         [0.75, 1, ],
#         [0.65, 1, ],
#         [0.6, 1, ],
#         [0.7, 1, ],
#         [0.7, 1, ],
#         [0.67, 1, ],
#         [0.78, 1, ],
#         [0.3, 0, ],
#         [0.4, 0, ],
#         [0.2, 0, ],
#     ]
# )

# X, y = a[:,0][:, np.newaxis], a[:,1][:, np.newaxis]

# oversample = SMOTE(sampling_strategy={0:3, 1:20}, random_state=random_state)

# X, y = oversample.fit_resample(X, y)

# print(X, y)

# X, y = a[:,0][:, np.newaxis], a[:,1][:, np.newaxis]

# oversample = ADASYN(sampling_strategy={0:3, 1:20}, random_state=random_state)

# X, y = oversample.fit_resample(X, y)

# print(X, y)