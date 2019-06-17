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
"""

ROW_SELECTION_PREFIX = "rs__"
CSV_SUFFIX = ".csv"

DIR_PROCESSED_DATA = "processed_data"
DIR_ROW_SELECTED = "row_selected_scales"

def check(input_dir_path, verbose=False, extra=False):
    output_dir_path = input_dir_path + "/" + DIR_PROCESSED_DATA
    output_row_selected_dir_path = output_dir_path + "/" + DIR_ROW_SELECTED + "/"

    nonuniques = []

    for filename in os.listdir(input_dir_path):
        if "rs__" not in filename:
            continue

        curr_scale_path = input_dir_path + "/" + filename

        print("*************************************************************")
        print("Handling scale = ", filename)

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, skiprows=[1])

        # Check if subjectkeys are unique
        if scale_df.shape[0] != len(scale_df["subjectkey"].unique()):
            nonuniques.append(filename)

        print(scale_df['subjectkey'].value_counts()[:10])

    print("\nScales with nonunique subjectkeys:")
    print(nonuniques)


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        check(sys.argv[1])
    else:
        raise Exception("Enter valid arguments\n"
              "\t path: the path to a real directory\n"
              "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3")