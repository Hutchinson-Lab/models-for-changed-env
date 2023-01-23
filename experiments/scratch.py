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

# import math

# print(math.pow(3,math.log(2,3)))


# def ss(X, Y):
#     m = len(X)
#     n = len(Y)
#     L = [[0 for i in range(n+1)] for j in range(m+1)]
 
#     # Following steps build L[m+1][n+1] in bottom up fashion. Note
#     # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
#     for i in range(m+1):
#         for j in range(n+1):
#             if i == 0 or j == 0:
#                 L[i][j] = 0
#             elif X[i-1] == Y[j-1]:
#                 L[i][j] = L[i-1][j-1] + 1
#             else:
#                 L[i][j] = max(L[i-1][j], L[i][j-1])
 
#         # Create a string variable to store the lcs string
#     ss = ""
 
#     # Start from the right-most-bottom-most corner and
#     # one by one store characters in lcs[]
#     i = m
#     j = n
#     while i > 0 and j > 0:
        
#         # If current character in X[] and Y are same, then
#         # current character is part of LCS
#         if X[i-1] == Y[j-1]:
#             ss += X[i-1]
#             i -= 1
#             j -= 1
 
#         # If not same, then find the larger of two and
#         # go in the direction of larger value
#         elif L[i-1][j] > L[i][j-1]:
#             ss += X[i-1]
#             i -= 1
             
#         else:
#             ss += Y[j-1]
#             j -= 1

#         # print(i, j, ss)
    
#     if i >= 1:  
#         while i > 0:
#             ss += X[i-1]
#             i -= 1

#     if j >= 1:
#         while j > 0:
#             ss += Y[j-1]
#             j -= 1

#     # We traversed the table in reverse order
#     # LCS is the reverse of what we got
#     ss = ss[::-1]
#     print("SS of " + X + " and " + Y + " is " + ss)

# pattern_1 = "BANANA"
# pattern_2 = "AYYAYYA"
# ss(pattern_1, pattern_2)

# from binarytree import build

# def longest_path (tree):
#     n = len(tree)
#     L = [1] * n
#     # node_list = list(root)

#     for i in range(1,n):

#         if (tree[i]!= None and tree[(i-1)//2] != None): # Check if parent is valid
#             L[i] = max(L[i], L[(i-1)//2]+L[i])
      
#         print(i, L)

#     # Find max in array
#     return max(L)

  
  
# # List of nodes
# #its left child at index 2*i+1, the right child at index 2*i+2 and parent at (i – 1)//2
# nodes =[3, 6, 8, 2, 11, None, 13] 
  
# # Building the binary tree
# binary_tree = build(nodes)

# print(longest_path(binary_tree.values))




# def largest_area_in_histogram (H):

#     max_area = 0
#     for i in range(len(H)): # left pointer
#         for j in range(len(H)): # right pointer

#             min_height = float('inf')
#             for k in range(i, j + 1):
#                 min_height = min(min_height, H[k])
            
#             max_area = max(max_area, min_height * (j - i + 1))
    
#     return max_area


# def maxRectangle(A):

#     # Calculate area for first row and
#     # initialize it as result
#     result = largest_area_in_histogram(A[0])# takes O(n^2)

#     # iterate over row to find maximum rectangular
#     # area considering each row as histogram
#     for i in range(1, len(A)): # takes O(n)

#         for j in range(len(A[i])): # takes O(n)

#             # if A[i][j] is 1 then add A[i -1][j]
#             if (A[i][j]):
#                 A[i][j] += A[i - 1][j]

#         # Update result if area with current
#         # row (as last row) of rectangle) is more
#         result = max(result, largest_area_in_histogram(A[i])) # takes O(n^2)

#     return result

# H = [2,1,5,6,2,3]

# print(largest_area_in_histogram(H))

# A = [[0, 1, 1, 0],
#          [1, 1, 1, 1],
#          [1, 1, 1, 1],
#          [1, 1, 0, 0]]

# print("Area of maximum rectangle is",
#         maxRectangle(A))

from scipy.stats import wasserstein_distance

arr1 = [[1,2],
        [1,2],
        [1,2],
        [1,2],]


arr2 = [[1,2],
        [1,2],
        [1,2],]
        

print(wasserstein_distance([0, 1, 3], [5, 6, 8,6]))

print(wasserstein_distance([5, 6, 8,6],[0, 1, 3],))

print(wasserstein_distance(arr1,arr2))
# print(wasserstein_distance([0, 1], [0, 1], [3, 1], [2, 2]))

# print(wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4],
                    #  [1.4, 0.9, 3.1, 7.2], [3.2, 3.5]))
