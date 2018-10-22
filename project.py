# -*- coding: utf-8 -*-
"""
CPSC 532M project
Joey
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

#make fake datasets
X, y = make_classification(n_samples=3000, n_features=700, n_informative=25, n_clusters_per_class=2)
X, norms = normalize(X, axis=0, return_norm=True)
X = X+1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#k-means clustering selecting features
kmeans = KMeans(n_clusters=50, random_state=0).fit(X_train.T)
features = pairwise_distances_argmin(kmeans.cluster_centers_, X_train.T)
selectedFeatures = X_train[:,features]

#calculating p-value according to chi-square distribution
chi2, pval = chi2(selectedFeatures, y_train)
seed = np.argpartition(pval, 25) #selecting top 25 feature
topFeatures = features[seed[:25]]

#subsampling
'''
TO DO
'''

#random forests with top 25 features
X_train_topFeature = X_train[:,topFeatures]
model = RandomForestClassifier(n_estimators=30)
model.fit(X_train_topFeature, y_train)
print("Training accuracy:", np.sum(model.predict(X_train_topFeature)==y_train)/y_train.shape[0])
print("Testing accuracy:", np.sum(model.predict(X_test[:,topFeatures])==y_test)/y_test.shape[0])

#Xgboost with top 25 features
dtrain = xgb.DMatrix(X_train_topFeature, label=y_train)
param = {'eval_metric': 'error'}
dtest = xgb.DMatrix(X_test[:,topFeatures], label=y_test)
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, 10, evallist)