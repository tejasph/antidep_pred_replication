import numpy as np
import pandas as pd
import sklearn as sk

import os
import sys

import stard_globals as gls
from utils import *
from stard_preprocessing_globals import SCALES, ORIGINAL_SCALE_NAMES

""" 
This will take in multiple text files (representing psychiatric scales) and output multiple CSV files, at least for each scale read in.

This script handles:
    - Selecting columns of interest (and throwing out the rest)
    - One hot encoding of specific columns
"""

ROW_SELECTION_PREFIX = "rs__"
ONE_HOT_ENCODED_PREFIX = "rs__ohe__"
CSV_SUFFIX = ".csv"

DIR_PROCESSED_DATA = "processed_data"
DIR_ROW_SELECTED = "row_selected_scales"
DIR_ONE_HOT_ENCODED = "one_hot_encoded_scales"

"""
root_data_dir_path is the path to the root of the folder containing the original scales

"""
def check(root_data_dir_path, verbose=False, extra=False):
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_row_selected_dir_path = output_dir_path + "/" + DIR_ROW_SELECTED + "/"
    output_one_hot_encoded_dir_path = output_dir_path + "/" + DIR_ONE_HOT_ENCODED + "/"

    input_dir_path = output_row_selected_dir_path

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_one_hot_encoded_dir_path):
            os.mkdir(output_one_hot_encoded_dir_path)

        # This is the prefix for the row selected scales.
        if ROW_SELECTION_PREFIX not in filename:
            continue

        curr_scale_path = input_dir_path + "/" + filename

        print("*************************************************************")
        print("Handling scale = ", filename)

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, skiprows=[1])

        # select columns
        # read in scales dictionary and check for if one hot encoding key exists

        if filename == "rs__dm01_enroll.csv":
            pass
        elif filename == "rs__phx01.csv":
            pass
        elif filename == "rs__qids01_w0c.csv":
            pass
        elif filename == "rs_idsc01.csv":
            pass
        elif filename == "rs__ccv01_w2.csv":
            pass

        output_file_name = ROW_SELECTION_PREFIX + scale_name
        scale_df = drop_empty_columns(scale_df)
        scale_df.to_csv(output_row_selected_dir_path + output_file_name + CSV_SUFFIX)


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        check(sys.argv[1])
    else:
        raise Exception("Enter valid arguments\n"
              "\t path: the path to a real directory\n"
              "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3")