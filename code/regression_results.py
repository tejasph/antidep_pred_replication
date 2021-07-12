#Regression Results
from run_regression import RunRegRun
import os

"""
Run Experiment script
"""

if __name__ == "__main__":

    exp_name = "test_mae"

    # Makes sure not to overwrite any files
    if os.path.isdir("results/" + exp_name):
        raise Exception("Name already exists")
    else:
        out_path = "results/" + exp_name + "/"
        os.mkdir(out_path)

    runs = 10
    regressors = ["rf"]
    X_paths = ["X_train_norm_over"]
    y = "y_train"
    y_proxies = ["score_change"]
    test_data = False

    for regressor in regressors:
        for y_proxy in y_proxies:
            for X_path in X_paths:
                RunRegRun(regressor, X_path, y, y_proxy, out_path,  runs , test_data)

    # runs = 10
    # regressor = "rf"
    # X_path = "X_train_norm_over"
    # y = "y_train_over"


    # RunRegRun(regressor, X_path, y, runs)