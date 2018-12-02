# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import numpy as np

def TreeEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    path = 'data/Toy_Data_Cleaned.csv'
    # read data and chop the header
    X = np.genfromtxt(path,delimiter=',')
    X = X[1:,]
    
    # Split the data into training and testing data   
    X_train ,X_test = train_test_split(X,test_size=0.2)
    y_train = X_train[:,0]
    y_test = X_test[:,0]
    X_train = X_train[:,1:]
    X_test = X_test[:,1:]
    
    # Feature selection
    features = featureSelectionChi(X_train,y_train,10,25)
    
    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=30)
    
    # Train an ensemble of 30 decision trees
    clf = [None]*30
    for i in range(30):
        clf[i] = DecisionTreeClassifier(random_state=0,max_depth=5)
        clf[i].fit(training[i][:,features],label[i])
    
    # Prediction
    n = X_test.shape[0]
    prediction = np.zeros((30,n))
    for i in range(30):
        prediction[i,:] = clf[i].predict(X_test[:,features])
    
    pred = stats.mode(prediction)[0][0]
    print("Accuracy is:", sum(pred==y_test)/n)
    
        
TreeEnsemble()

