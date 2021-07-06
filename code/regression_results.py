#Regression Results
from run_regression import RunRegRun
import os

"""
Run Experiment script
"""

if __name__ == "__main__":

    exp_name = "overlap_resp_rem_stand"

    # Makes sure not to overwrite any files
    if os.path.isdir("results/" + exp_name):
        raise Exception("Name already exists")
    else:
        out_path = "results/" + exp_name + "/"
        os.mkdir(out_path)

    runs = 100
    regressor = "rf"
    X_path = "X_train_stand_over"
    y = "y_train_over"
    test_data = False

    RunRegRun(regressor, X_path, y, out_path,  runs , test_data)

    # runs = 10
    # regressor = "rf"
    # X_path = "X_train_norm_over"
    # y = "y_train_over"


    # RunRegRun(regressor, X_path, y, runs)