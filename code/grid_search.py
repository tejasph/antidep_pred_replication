# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:21:40 2020

@author: jjnun
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np

pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200311\1_Replication\X_lvl2_rem_qids01__final.csv'
pathLabel = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND_Replication\teyden-git\data\final_datasets\to_run_20200311\1_Replication\y_lvl2_rem_qids01__final.csv'

X = np.genfromtxt(pathData, delimiter=',')
y = np.genfromtxt(pathLabel, delimiter=',')[1:,1]
X = X[1:,1:]

"""
model = SGDClassifier(loss='log', penalty='elasticnet', max_iter=100)
grid = GridSearchCV(estimator=model, n_jobs=6, scoring='balanced_accuracy', param_grid={
    'l1_ratio': [p/1000 for p in range(650,705,5)],
    'max_iter': [10000],
    'alpha': [0.05,0.06,0.07,0.08,0.09, 0.1,0.11],
    'power_t':[0.005,0.08, 0.09,0.01,0.011, 0.012]})
grid.fit(X,y)
print(grid)
# summarize the results of the grid search
print(f' The best balanced accuracy: {grid.best_score_}')
print(f'The best l1_ratio {grid.best_estimator_.l1_ratio}')
print(f'The best max_iter {grid.best_estimator_.max_iter}')
print(f'The best alpha {grid.best_estimator_.alpha}')
print(f'The best power_t {grid.best_estimator_.power_t}')
"""

model = LogisticRegression(penalty='l2',solver='lbfgs', max_iter=10000)
grid = GridSearchCV(estimator=model, n_jobs=6, scoring='balanced_accuracy', param_grid={
    'C': [p/1000 for p in range(90, 120, 1)]})
    # 'max_iter': [1000, 10000, 20000]})
    #'solver': ['lbfgs', saga, liblinear]})

grid.fit(X,y)
print(grid)
# summarize the results of the grid search
print(f'The best balanced accuracy: {grid.best_score_:.4f}')
print(f'The best max_iter {grid.best_estimator_.max_iter}')
print(f'The best C {grid.best_estimator_.C}')
print(f'The best solver {grid.best_estimator_.solver}')