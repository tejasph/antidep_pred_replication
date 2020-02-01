"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np

def ultraEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    pathData = '../data/canbind-clean-aggregated-data.with-id.csv'
    pathLabel = '../data/targets.csv'
    # read data and chop the header
    X = np.genfromtxt(pathData, delimiter=',')
    y = np.genfromtxt(pathLabel, delimiter=',')[:,1]
    X = X[1:,]
    X = X[:,2:]
    
    n,m = X.shape
    kf = cross_validation.KFold(n, n_folds=10, shuffle=True)

    j=1
    accu = np.empty([10,], dtype=float)
    for train_index, test_index in kf:
        print("Fold:", j)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Feature selection
        #features = featureSelectionChi(X_train,y_train,15,25)
        features = featureSelectionELAS(X_train,y_train,15)
        #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),10)

        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=50)
 
        # Train an ensemble of 30 decision trees
        clf = [None]*50
        for i in range(10):
            clf[i] = SGDClassifier(loss='log', penalty='elasticnet', max_iter=50, alpha=0.01, l1_ratio=0.15)
            clf[i].fit(training[i][:,features],label[i])
        for i in range(10,20):
            clf[i] = RandomForestClassifier(n_estimators=50, n_jobs = 5)
            clf[i].fit(training[i][:,features],label[i])
        for i in range(20,30):
            clf[i] = MLPClassifier(solver='sgd',alpha=1e-3,hidden_layer_sizes=(50,)
            ,max_iter=2000,learning_rate='adaptive')
            #clf[i].fit(training[i][:,features],label[i])
            clf[i].fit(training[i][:,features],label[i])
        for i in range(30,40):
            clf[i] = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=10, random_state=0)
            clf[i].fit(training[i][:,features],label[i])
        for i in range(40,50):
            clf[i] = SVC(probability=True)
            clf[i].fit(training[i][:,features],label[i])

        # Prediction
        n = X_test.shape[0]
        # calculting the average probabilty of each class
        pred_prob = np.zeros((n,2))
        for i in range(50):
            pred_prob += clf[i].predict_proba(X_test[:,features])
            
        pred_prob = pred_prob/50
        # Pick the class with the greastest probability to be the prediction
        pred = np.argmax(pred_prob,axis=1)
        print(pred_prob)
        y_score = pred_prob[:,1]
    
        # Report accuracy and draw ROC curve
        drawROC(y_test, y_score)
        score = sum(pred==y_test)/n
        print("Accuracy is:", score)
        accu[j-1] = score
        j = j+1
    print("Average accuracy is:",sum(accu)/10)

ultraEnsemble()


