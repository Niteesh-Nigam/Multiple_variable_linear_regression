import numpy as np
import math, copy
import matplotlib.pyplot as plt

data = np.genfromtxt('./Admission_Predict.csv', delimiter=',', skip_header=1)
# print(data)
X_train_unnormalized= np.array(data[:,1:7])
y_train = np.array(data[:,8])

def zscore_normalize_features(X):
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return X_norm

def mean_normalize_features(X):
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    x_max     = np.max(X, axis=0)  
    x_min     = np.min(X, axis=0)  
    X_norm = (X - mu) / (x_max-x_min)   

    return X_norm

# X_train =X_train_unnormalized
X_train = mean_normalize_features(X_train_unnormalized)
print(X_train)