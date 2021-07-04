#Regression Results
from run_regression import RunRegRun

"""
Temporary
"""

if __name__ == "__main__":
    runs = 10
    regressor = "rf"
    X_path = "X_train_norm"
    y = "y_train"
    test_data = True

    RunRegRun(regressor, X_path, y, runs, test_data)

    # runs = 10
    # regressor = "rf"
    # X_path = "X_train_norm_over"
    # y = "y_train_over"


    # RunRegRun(regressor, X_path, y, runs)