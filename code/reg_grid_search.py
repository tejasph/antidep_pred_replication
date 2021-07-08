#reg_grid_search.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
import pandas as pd
import numpy as np
import datetime
import pickle
import os


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

    return model, params

def select_target(y, y_proxy):
    if y_proxy == "score_change":
        y_target = y[['target_change']]
    elif y_proxy == "final_score":
        y_target = y[['target_score']]

    return y_target

def optimize_params(model, params, X, y, filename):
    startTime = datetime.datetime.now()

    search = HalvingRandomSearchCV(model, params, cv = 10, scoring = 'neg_root_mean_squared_error', n_jobs = -1, verbose = 1)

    search.fit(X, y.to_numpy().ravel())

    print(search.best_score_)
    print(search.best_estimator_)

    f = open(os.path.join("results/optimized_params", filename + '.txt'), 'w')
    f.write("Best estimator RMSE is: {:.4f}\n".format(search.best_score_))
    f.write("The best performing model is: {}".format(search.best_estimator_))

    pickle.dump(search.best_estimator_, open("results/optimized_params/" + filename + ".pkl", 'wb'))
    
    print("Completed after seconds: \n")
    print(datetime.datetime.now() - startTime)

    return

if __name__== "__main__":



    X_train = pd.read_csv("data/modelling/X_train_norm.csv").set_index('subjectkey')
    y_train = pd.read_csv("data/modelling/y_train.csv").set_index('subjectkey')

    regressors = ['rf']
    y_proxies = ['score_change']
    X_type = ['X_train_norm']

    for reg in regressors:
        for y_proxy in y_proxies:
            
            filename = "{}_{}".format(reg, y_proxy)
            print(filename)
            model, params = select_model_and_params(reg)
            y_target = select_target(y_train, y_proxy)
            print(y_target)
            optimize_params(model, params, X_train, y_target, filename)



    # Optimizing for Rf models

    # optimize_params(rf_model, rf_params, X, y_score_change)
    # optimize_params(rf_model, rf_params, X, y_final_score)

