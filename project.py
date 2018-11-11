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
from astropy.stats import bootstrap as astro_bootstrap
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
TO DO, implement the bootstrapping 
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

##Helper Functions -

def bootstrap(X,n):
    """ Produce one boostrap sample from X of n example. Uses astropy's boostrap """
    bootsample = astro_bootstrap(X,bootnum=1,samples=n)
    return bootsample[0,:,:]

# subsample
# Input: X n,d data matrix, i int      

def subsample(X,c,t=30) :
    """ Produces subsamples as from the paper, using boostraps from the majority class
    (the class with more examples) and all of the minority class divided into t subsamples
    Arguments:
        X -- Data matrix of n,d dimensions
        c -- Column of X to sort X into majority and minority class, must be binary (1 or 0)
        t -- How many subsamples to produce; in the Zie paper, they used 30
    Returns:
        Matrix of dimensions (t,a,d) where a is the number of samples in each subsample
    """
    n, d = X.shape
    
    # TO_DO: Remove this when final data acquired
    # Makes the target column into binary data to simulate final data
    X[:,c] = X[:,c] > 0.981

    count_1 = np.sum(X[:,c] == 1)
    count_0 = np.sum(X[:,c] == 0)
    
    # Set majority and minority classes
    if count_1 > count_0:
        major = 1
        minor = 0
        count_minor = count_0
    else:
        major = 0
        minor = 1
        count_minor = count_1
    
    # throw exception if there are data entries not 1 or 0 in this column
    assert ((count_1 + count_0) == n), "Column to decide classes has non 0 or 1 entries; must be binary"
    
    # Create matrices of only majority and minority examples
    X_major = X[X[:,c] == major] # Samples with values of the majority in coloumn c
    X_minor = X[X[:,c] == minor] # "" but minority
    
    # TO_DO: Remove this when final data acquireed
    # Remove last entries of X_minor so divisible by t
    correction = count_minor % t
    X_minor = X_minor[:-correction,:]
    count_minor = count_minor - correction

    # Ensures the number of examples in the minor class are divisible by t
    assert ((count_minor % t) == 0), "Number of samples in the minor class must be divisible by t: " + str(count_minor % t)
    n_per_t = int(count_minor/t)
    
    # Loop to make the t subsamples
    subsamples = np.zeros((t,2*n_per_t,d))
    for i in range(t):
        #Create empty array
        subsample = np.zeros((2*n_per_t,d))
        # Add into boostraps from majority examples
        subsample[0:n_per_t,:] = bootstrap(X_major,n_per_t)
        # Add in examples from minority examples (not bootstrapped)
        subsample[n_per_t:,:] = X_minor[i*n_per_t:(i+1)*n_per_t,:]
        # Shuffle data in case this matters
        subsample = np.random.shuffle(subsample)
        
        subsamples[i,:,:] = subsample 
    
    return subsamples


print("Here's an example of using subsample, it has shape: " + str(subsample(X,0,10).shape))