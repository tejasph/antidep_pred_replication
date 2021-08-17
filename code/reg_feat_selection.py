# Feature Selection
import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from run_globals import REG_MODEL_DATA_DIR
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet


# Helpful article https://heartbeat.fritz.ai/feature-ranking-with-recursive-feature-elimination-3e22db639208

def RunFeatSelection(selection_method, X_train_path, y_train_path, X_test_path):

    # Set up paths
    X_path = os.path.join(REG_MODEL_DATA_DIR, X_train_path + ".csv")
    y_path = os.path.join(REG_MODEL_DATA_DIR, y_train_path + ".csv")
    X_test_path = os.path.join(REG_MODEL_DATA_DIR, X_test_path + ".csv")

    X = pd.read_csv(X_path).set_index('subjectkey')
    X_test = pd.read_csv(X_test_path).set_index('subjectkey')
    y = pd.read_csv(y_path).set_index('subjectkey')
    print(X.shape)

    # Feature selection module
    if selection_method == "rfe":
        base_model = ElasticNet(l1_ratio = 1)
        rfe_model = RFECV(base_model, step = 1, verbose = 2, n_jobs = -1)
        rfe_model.fit(X,y)

        # model = RandomForestRegressor()
        # pipe = Pipeline([('Feature Selection', rfe_model), ('model', model)])
        # cv = KFold(n_splits=10)
        # n_scores = cross_val_score(pipe, X, y, scoring = 'neg_root_mean_squared_error', cv = cv, n_jobs = -1)
        # print(np.mean(n_scores))

        # pipe.fit(X, y)

        print(rfe_model.n_features_)
        ranks = pd.DataFrame(rfe_model.ranking_, index = X.columns, columns = ['Rank'])
        print(ranks.sort_values(by = 'Rank', ascending = True).iloc[0:60, :])

        X.iloc[:,rfe_model.get_support(indices = True)].to_csv("data/modelling/X_train_norm_select.csv", index = True)
        X_test.iloc[:,rfe_model.get_support(indices = True)].to_csv("data/modelling/X_test_norm_select.csv", index = True)

        # print(rfe_model.get_support(indices = True))
        # print(X.iloc[:, rfe_model.get_support(indices = True)])

        # plt.figure()
        # plt.xlabel("Number of features selected")
        # plt.ylabel("Cross validation score (R_square)")
        # plt.plot(len(rfe_model.grid_scores_), rfe_model.grid_scores_)
        # plt.savefig("rfe_CV.png", bbox_inches = 'tight')
        # plt.close()


if __name__ == "__main__":
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])


    RunFeatSelection(sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4])