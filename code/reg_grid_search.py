#reg_grid_search.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
import datetime
import pickle
import os

def select_X(X_type):

    if X_type == "X_train_norm":
        X_train = pd.read_csv("data/modelling/X_train_norm.csv").set_index('subjectkey')
    elif X_type == "X_train_norm_over":
        X_train = pd.read_csv("data/modelling/X_train_norm_over.csv").set_index('subjectkey')

    return X_train

def select_model_and_params(model_type):
    if model_type == "rf":
        model = RandomForestRegressor(n_jobs = -1)
        max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
        max_depth.append(None)
        params = {
                    'max_features': ['auto', 'sqrt', 'log2', 0.33],
                    'max_depth': max_depth,
                    'min_samples_split': np.arange(2,15),
                    'min_samples_leaf': np.arange(2,15),
                    'bootstrap':  [True, False],
                        'criterion':['mse','mae'],
                        'max_samples':[0.7,0.8,0.9, None]}
    elif model_type == "gbdt":
        model = GradientBoostingRegressor(n_iter_no_change = 10)
        max_depth = [int(x) for x in np.arange(2,15)]
        max_depth.append(None)

        params = {
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'max_depth': max_depth,
            'min_samples_split': np.arange(2,15),
            'min_samples_leaf': np.arange(2, 15),
            'max_features': ['auto', 'sqrt', 'log2', 0.33],
            'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    
    

    return model, params

def select_target(y, y_proxy):
    if y_proxy == "score_change":
        y_target = y[['target_change']]
    elif y_proxy == "final_score":
        y_target = y[['target_score']]

    return y_target

def optimize_params(model, params, X, y, filename):
    

    search = HalvingRandomSearchCV(model, params, cv = 10, scoring = 'neg_root_mean_squared_error', n_jobs = -1, verbose = 1)

    search.fit(X, y.to_numpy().ravel())

    print(search.best_score_)
    print(search.best_estimator_)

    # f = open(os.path.join("results/optimized_params", filename + '.txt'), 'w')
    # f.write("Best estimator RMSE is: {:.4f}\n".format(search.best_score_))
    # f.write("The best performing model is: {}".format(search.best_estimator_))

    # pickle.dump(search.best_estimator_, open("results/optimized_params/" + filename + ".pkl", 'wb'))

    return search.best_estimator_, search.best_score_

if __name__== "__main__":

    runs = 30
    score_dict = {'run':[], 'model':[], 'best_score':[]}

    # X_train = pd.read_csv("data/modelling/X_train_norm.csv").set_index('subjectkey')
    y_train = pd.read_csv("data/modelling/y_train.csv").set_index('subjectkey')

    regressors = ['gbdt']
    y_proxies = ['score_change', 'final_score']
    X_types = ['X_train_norm_over']

    for reg in regressors:
        for y_proxy in y_proxies:
            for X_type in X_types:

                startTime = datetime.datetime.now()
                filename = "{}_{}_{}".format(reg, X_type, y_proxy)
                print(filename)
                model, params = select_model_and_params(reg)
                X_train = select_X(X_type)
                y_target = select_target(y_train, y_proxy)

                print(y_target)
                for r in range(runs):
                    best_estimator, best_score = optimize_params(model, params, X_train, y_target, filename)
                    score_dict['run'].append(r)
                    score_dict['model'].append(best_estimator)
                    score_dict['best_score'].append(best_score)
                
                score_df = pd.DataFrame(score_dict).sort_values(by = ['best_score'], ascending = False)
                print(score_df)
                print(score_df.iloc[0,1])
                print(score_df.iloc[0,2])
                f = open(os.path.join("results/optimized_params", filename + '.txt'), 'w')
                f.write("Best estimator RMSE is: {:.4f}\n".format(score_df.iloc[0,2]))
                f.write("The best performing model is: {}".format(score_df.iloc[0,1]))

                pickle.dump(score_df.iloc[0,1], open("results/optimized_params/" + filename + ".pkl", 'wb'))
                score_df.to_csv("results/optimized_params/" + filename + ".csv", index = False)

                print("Completed optimization after seconds: \n")
                print(datetime.datetime.now() - startTime)

    # Optimizing for Rf models

    # optimize_params(rf_model, rf_params, X, y_score_change)
    # optimize_params(rf_model, rf_params, X, y_final_score)

# RandomForestRegressor(max_depth=30, max_samples=0.8, min_samples_leaf=6,
#                       min_samples_split=11, n_jobs=-1)

# RandomForestRegressor(max_samples=0.7, min_samples_leaf=4, min_samples_split=8,
#                       n_jobs=-1)