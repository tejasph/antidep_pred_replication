import os
import sys
import pandas as pd
import numpy as np
from collections import namedtuple

import stard_globals as gls
from utils import *

""" 
This will take in multiple text files (representing psychiatric scales) and output multiple CSV files, at least for each scale read in.
"""

SCALES = {
    "dm01": {
    },
    "ccv01": {
    },
    "crs01": {
    },
    "hrsd01": {
    },
    "idsc01": {
    },
    "mhx01": {
    },
    "pdsq01": {
    },
    "phx01": {
    },
    "qids01": {
    },
    "qlesq01": {
    },
    "sfhs01": {
    },
    "side_effects01": {
    },
    "ucq01": {
    },
    "wpai01": {
    },
    "wsas01": {
    }
}

ROW_SELECTION_PREFIX = "rs__"
CSV_SUFFIX = ".csv"

def select_rows(input_dir_path, verbose=False, extra=False):
    output_dir_path = input_dir_path + "/processed_data"
    output_row_selected_dir_path = output_dir_path + "/row_selected_scales/"

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
            if not os.path.exists(output_row_selected_dir_path):
                os.mkdir(output_row_selected_dir_path)

        scale_name = filename.split(".")[0]
        if scale_name not in SCALES:
            continue

        curr_scale_path = input_dir_path + "/" + filename

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, sep='\t', skiprows=[1])
        scale_df = drop_empty_columns(scale_df)

        print("*************************************************************")
        print("Handling scale = ", scale_name)

        if scale_name in ["ccv01", "side_effects01"]:
            if scale_df["week"].isnull().values.any():
                raise Exception("Numerical column should not contain any null values.")

            # Convert column to float type
            scale_df.loc[:, "week"] = scale_df["week"].astype("float")

            # Split into 2 separate files
            if scale_name == "side_effects01":
                # This scale has the values in the level column as floats.
                criteria_1_df = scale_df[(scale_df["level"] == 1) & (scale_df["week"] < 1)]
                criteria_2_df = scale_df[(scale_df["level"] == 1) & (2 <= scale_df["week"]) & (scale_df["week"] < 3)]
            else:
                criteria_1_df = scale_df[(scale_df["level"] == "Level 1") & (scale_df["week"] < 1)]
                criteria_2_df = scale_df[(scale_df["level"] == "Level 1") & (2 <= scale_df["week"]) & (scale_df["week"] < 3)]

            output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "_w0"
            output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w2"

            criteria_1_df = drop_empty_columns(criteria_1_df)
            criteria_2_df = drop_empty_columns(criteria_2_df)

            criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX)
            criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX)

        elif scale_name == "dm01":
            scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("int")

            criteria_1_df = scale_df[scale_df["level"] == "Enrollment"]
            criteria_2_df = scale_df[(scale_df["level"] == "Level 1") & (scale_df["days_baseline"] < 7)]

            output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "enroll"
            output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "w0"

            criteria_1_df = drop_empty_columns(criteria_1_df)
            criteria_2_df = drop_empty_columns(criteria_2_df)

            criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX)
            criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX)

        elif scale_name == "qids01":
            # Starts with 84,932 rows of which 39,380 are null for column "week"
            print("Number of qids week column are null before replacing with week values matching key (subjectkey, "
                  "days_baseline, level): {}".format(sum(scale_df["week"].isnull()))) # 39380 are null
            Entry = namedtuple("Entry", ["subjectkey", "days_baseline", "level"])
            tracker = {}
            missed = {}

            # Add all non-blanks to dictionary
            week_nonnull_scale_df = scale_df[scale_df["week"].notnull()]
            for idx, row in week_nonnull_scale_df.iterrows():
                entry = Entry(row["subjectkey"], row["days_baseline"], row["level"])
                tracker[entry] = row["week"]

            # Replace all blanks for "week" with the value of a matching entry key
            week_null_scale_df = scale_df[scale_df["week"].isnull()]
            for idx, row in week_null_scale_df.iterrows():
                entry = Entry(row["subjectkey"], row["days_baseline"], row["level"])
                if entry in tracker:
                    scale_df.loc[row.name, "week"] = tracker[entry]
                else:
                    missed[entry] = row["week"]

            print("Number of qids week column are null after replacing with week values matching key (subjectkey, "
                  "days_baseline, level): {}".format(sum(scale_df["week"].isnull()))) # 13234 are null
            print("Number of qids rows before eliminating rows empty for all of {} columns: {}"
                  .format(["vsoin", "vmnin", "vemin", "vhysm", "vmdsd"], scale_df.shape[0]))
            scale_df = scale_df[(scale_df["vsoin"].notnull())
                                & (scale_df["vmnin"].notnull())
                                & (scale_df["vemin"].notnull())
                                & (scale_df["vhysm"].notnull())
                                & (scale_df["vmdsd"].notnull())]
            print("Number of qids rows after eliminating rows empty for all of {} columns: {}"
                  .format(["vsoin", "vmnin", "vemin", "vhysm", "vmdsd"], scale_df.shape[0]))

            output_file_name = "pre" + ROW_SELECTION_PREFIX + "pre" + scale_name
            scale_df = drop_empty_columns(scale_df)
            scale_df.to_csv(output_row_selected_dir_path + output_file_name + CSV_SUFFIX)

        else:
            if scale_name in ["mhx01", "pdsq01", "phx01"]:
                scale_df = scale_df.drop_duplicates(subset='subjectkey', keep='first')
            elif scale_name in ["qlesq01", "sfhs01", "ucq01", "wpai01", "wsas01"]:
                scale_df = scale_df[scale_df["CallType"] == "Base"]
            elif scale_name == "crs01":
                scale_df = scale_df[scale_df["crcid"].notnull()]
            elif scale_name == "hrsd01":
                scale_df = scale_df[scale_df["level"] == "Enrollment"]
            elif scale_name == "idsc01":
                scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("int")
                scale_df = scale_df[(scale_df["level"] == "Level 1") & (scale_df["days_baseline"] < 8)]
            output_file_name = ROW_SELECTION_PREFIX + scale_name
            scale_df = drop_empty_columns(scale_df)
            scale_df.to_csv(output_row_selected_dir_path + output_file_name + CSV_SUFFIX)

    # Handle preqids, after looping through the original scales
    preqids_file_path = output_row_selected_dir_path + "prers__preqids01.csv"
    if os.path.exists(preqids_file_path):
        scale_df = pd.read_csv(preqids_file_path)
        scale_df = scale_df.drop(columns=["Unnamed: 0"])

        # Convert column to float type
        scale_df.loc[:, "week"] = scale_df["week"].astype("float")

        # Split into 3 separate files
        criteria_1_df = scale_df[(scale_df["level"] == "Level 1")
                                 & (scale_df["week"] < 1)
                                 & (scale_df["version_form"] == "Clinician")]
        criteria_2_df = scale_df[(scale_df["level"] == "Level 1")
                                 & (scale_df["week"] < 1)
                                 & (scale_df["version_form"] == "Self Rating")]
        criteria_3_df = scale_df[(scale_df["level"] == "Level 1")
                                 & (2 <= scale_df["week"]) & (scale_df["week"] < 3)
                                 & (scale_df["version_form"] == "Clinician")]
        criteria_4_df = scale_df[(scale_df["level"] == "Level 1")
                                 & (2 <= scale_df["week"])
                                 & (scale_df["week"] < 3)
                                 & (scale_df["version_form"] == "Self Rating")]

        criteria_1_df = drop_empty_columns(criteria_1_df)
        criteria_2_df = drop_empty_columns(criteria_2_df)
        criteria_3_df = drop_empty_columns(criteria_3_df)
        criteria_4_df = drop_empty_columns(criteria_4_df)

        scale_name = "qids01"
        output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "_w0c"
        output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w0sr"
        output_file_name_3 = ROW_SELECTION_PREFIX + scale_name + "_w2c"
        output_file_name_4 = ROW_SELECTION_PREFIX + scale_name + "_w2sr"

        criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX)
        criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX)
        criteria_3_df.to_csv(output_row_selected_dir_path + output_file_name_3 + CSV_SUFFIX)
        criteria_4_df.to_csv(output_row_selected_dir_path + output_file_name_4 + CSV_SUFFIX)

def drop_empty_columns(df):
    return df.dropna(axis="columns", how="all")  # Drop columns that are all empty

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        select_rows(sys.argv[1])
    else:
        raise Exception("Enter valid arguments\n"
              "\t path: the path to a real directory\n"
              "\t e.g. python stard_row_selector.py /Users/teyden/Downloads/stardmarch19v3")
