import os
import sys
import pandas as pd
import numpy as np
from collections import namedtuple

from utils import *
from stard_preprocessing_globals import ORIGINAL_SCALE_NAMES, SCALES, VALUE_CONVERSION_MAP

""" 
This will take in multiple text files (representing psychiatric scales) and output multiple CSV files, at least for each scale read in.
"""

ROW_SELECTION_PREFIX = "rs__"
COLUMN_SELECTION_PREFIX = ROW_SELECTION_PREFIX + "cs__"
ONE_HOT_ENCODED_PREFIX = COLUMN_SELECTION_PREFIX + "ohe__"
VALUES_CONVERTED_PREFIX = ONE_HOT_ENCODED_PREFIX + "vc__"
AGGREGATED_ROWS_PREFIX = VALUES_CONVERTED_PREFIX + "ag__" # Final: "rs__cs__ohe__vc__ag__" which represents the order of the pipeline
CSV_SUFFIX = ".csv"

DIR_PROCESSED_DATA = "processed_data"
DIR_ROW_SELECTED = "row_selected_scales"
DIR_COLUMN_SELECTED = "column_selected_scales"
DIR_ONE_HOT_ENCODED = "one_hot_encoded_scales"
DIR_VALUES_CONVERTED = "values_converted_scales"
DIR_AGGREGATED_ROWS = "aggregated_rows_scales"

def select_rows(input_dir_path):
    output_dir_path = input_dir_path + "/" + DIR_PROCESSED_DATA
    output_row_selected_dir_path = output_dir_path + "/" + DIR_ROW_SELECTED + "/"

    print("\n--------------------------------1. ROW SELECTION-----------------------------------\n")

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_row_selected_dir_path):
            os.mkdir(output_row_selected_dir_path)

        scale_name = filename.split(".")[0]
        if scale_name not in ORIGINAL_SCALE_NAMES:
            continue

        # if scale_name != "qlesq01":
        #     continue

        curr_scale_path = input_dir_path + "/" + filename

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, sep='\t', skiprows=[1])
        scale_df = drop_empty_columns(scale_df)

        print("*************************************************************")
        print("Handling scale = ", scale_name)

        selection_criteria = ORIGINAL_SCALE_NAMES[scale_name]

        if scale_name in ["ccv01"]:
            if scale_df["week"].isnull().values.any():
                raise Exception("Numerical column should not contain any null values.")

            # Convert column to float type
            scale_df.loc[:, "week"] = scale_df["week"].astype("float")

            # Split into 2 separate files
            # if scale_name == "side_effects01":
            #     # This scale has the values in the level column as floats.
            #     criteria_1_df = scale_df[(scale_df["level"] == 1) & (scale_df["week"] < 3)]
            #     criteria_2_df = scale_df[(scale_df["level"] == 1) & (2 <= scale_df["week"]) & (scale_df["week"] < 3)]
            if scale_name == "ccv01":
                # criteria_1_df = scale_df[(scale_df["level"] == "Level 1") & (scale_df["week"] == 2)]
                criteria_2_df = scale_df[(scale_df["level"] == "Level 1") & (2 <= scale_df["week"]) & (scale_df["week"] < 3)]

            # output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "_w0"
            output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w2"

            # criteria_1_df = select_subject_rows(criteria_1_df, scale_name, selection_criteria)
            criteria_2_df = select_subject_rows(criteria_2_df, scale_name, selection_criteria)

            # criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX, index=False)
            criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX, index=False)

        elif scale_name == "dm01":
            scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("int")

            criteria_1_df = scale_df[scale_df["level"] == "Enrollment"]
            criteria_2_df = scale_df[(scale_df["level"] == "Level 1") & (scale_df["days_baseline"] < 7)]

            output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "_enroll"
            output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w0"

            criteria_1_df = select_subject_rows(criteria_1_df, scale_name, selection_criteria)
            criteria_2_df = select_subject_rows(criteria_2_df, scale_name, selection_criteria)

            criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX, index=False)
            criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX, index=False)

        # Handles creating the preliminary file. See end of this function to see how qids was split up.
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
            scale_df.to_csv(output_row_selected_dir_path + output_file_name + CSV_SUFFIX, index=False)

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
                scale_df = scale_df[(scale_df["level"] == "Level 1") & (scale_df["days_baseline"] < 15) & (scale_df["time_point"] == 1)]
            elif scale_name == "side_effects01":
                scale_df.loc[:, "week"] = scale_df["week"].astype("float")
                scale_df = scale_df[(scale_df["level"] == 1) & (scale_df["week"] < 3)]

            output_file_name = ROW_SELECTION_PREFIX + scale_name
            scale_df = select_subject_rows(scale_df, scale_name, selection_criteria)
            scale_df.to_csv(output_row_selected_dir_path + output_file_name + CSV_SUFFIX, index=False)

    # Handle preqids, after looping through the original scales
    preqids_file_path = output_row_selected_dir_path + "prers__preqids01.csv"
    if os.path.exists(preqids_file_path):
        scale_df = pd.read_csv(preqids_file_path)
        # scale_df = scale_df.drop(columns=["Unnamed: 0"])

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

        scale_name = "qids01"
        selection_criteria = ORIGINAL_SCALE_NAMES[scale_name]
        criteria_1_df = select_subject_rows(criteria_1_df, scale_name, selection_criteria)
        criteria_2_df = select_subject_rows(criteria_2_df, scale_name, selection_criteria)
        criteria_3_df = select_subject_rows(criteria_3_df, scale_name, selection_criteria)
        criteria_4_df = select_subject_rows(criteria_4_df, scale_name, selection_criteria)

        output_file_name_1 = ROW_SELECTION_PREFIX + scale_name + "_w0c"
        output_file_name_2 = ROW_SELECTION_PREFIX + scale_name + "_w0sr"
        output_file_name_3 = ROW_SELECTION_PREFIX + scale_name + "_w2c"
        output_file_name_4 = ROW_SELECTION_PREFIX + scale_name + "_w2sr"

        criteria_1_df.to_csv(output_row_selected_dir_path + output_file_name_1 + CSV_SUFFIX, index=False)
        criteria_2_df.to_csv(output_row_selected_dir_path + output_file_name_2 + CSV_SUFFIX, index=False)
        criteria_3_df.to_csv(output_row_selected_dir_path + output_file_name_3 + CSV_SUFFIX, index=False)
        criteria_4_df.to_csv(output_row_selected_dir_path + output_file_name_4 + CSV_SUFFIX, index=False)

def select_subject_rows(scale_df, scale_name, selection_criteria):
    if selection_criteria == {}:
        return scale_df

    selector_col_name = selection_criteria["subjectkey_selector"]
    preference = selection_criteria["preference"]
    subject_group = scale_df.groupby(["subjectkey"])

    for subjectkey, subject_rows_df in subject_group:
        condition_val = np.nanmin(subject_rows_df[selector_col_name])
        if preference == "larger":
            condition_val = np.nanmax(subject_rows_df[selector_col_name])

        if condition_val is np.nan:
            print("NaN:", subjectkey, scale_name)

        # There could be multiple matches
        matches = scale_df[(scale_df["subjectkey"] == subjectkey) & (scale_df[selector_col_name] == condition_val)]
        if len(matches) > 0:
            # print(scale_name, selector_col_name, subjectkey, condition_val, subject_rows_df[selector_col_name])
            scale_df = scale_df[(scale_df["subjectkey"] != subjectkey) | (scale_df.index == matches.index[0])]
        else:
            # print(scale_name, selector_col_name, subjectkey, condition_val, subject_rows_df[selector_col_name])
            scale_df = scale_df[(scale_df["subjectkey"] != subjectkey)]

    return scale_df

"""
root_data_dir_path is the path to the root of the folder containing the original scales
"""
def select_columns(root_data_dir_path):
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_row_selected_dir_path = output_dir_path + "/" + DIR_ROW_SELECTED + "/"
    output_column_selected_dir_path = output_dir_path + "/" + DIR_COLUMN_SELECTED + "/"

    input_dir_path = output_row_selected_dir_path

    print("\n--------------------------------2. COLUMN SELECTION-----------------------------------\n")

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_column_selected_dir_path):
            os.mkdir(output_column_selected_dir_path)

        # This is the prefix for the row selected scales.
        if "rs" != filename.split("__")[0]:
            continue

        curr_scale_path = input_dir_path + "/" + filename

        scale_name = filename.split(".")[0].split("__")[-1]
        print("*************************************************************")
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, skiprows=[1])

        # Drop empty columns
        scale_df = drop_empty_columns(scale_df)

        whitelist = SCALES[scale_name]["whitelist"]

        # Add subject key so that you know which subject it is
        if scale_name != "qids01_w0c":
            whitelist.append("subjectkey")

        # Select columns in the whitelist
        scale_df = scale_df[whitelist]

        output_file_name = COLUMN_SELECTION_PREFIX + scale_name
        scale_df.to_csv(output_column_selected_dir_path + output_file_name + CSV_SUFFIX, index=False)

def one_hot_encode_scales(root_data_dir_path):
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_column_selected_dir_path = output_dir_path + "/" + DIR_COLUMN_SELECTED + "/"
    output_one_hot_encoded_dir_path = output_dir_path + "/" + DIR_ONE_HOT_ENCODED + "/"

    input_dir_path = output_column_selected_dir_path

    print("\n--------------------------------3. ONE HOT ENCODING-----------------------------------\n")

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_one_hot_encoded_dir_path):
            os.mkdir(output_one_hot_encoded_dir_path)

        # This is the prefix for the row, then column selected scales.
        if "rs__cs__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        cols_to_one_hot_encode = []
        if "one_hot_encode" in SCALES[scale_name]:
             cols_to_one_hot_encode = SCALES[scale_name]["one_hot_encode"]

        print("*************************************************************")
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file
        scale_df = pd.read_csv(input_dir_path + "/" + filename, skiprows=[1])

        if scale_name == "dm01_enroll":
            cols_to_convert = ['empl', 'volun', 'leave', 'publica', 'medicaid', 'privins']
            conversion_map = {15: np.nan, 9: np.nan, -7: np.nan}
            for col_name in cols_to_convert:
                scale_df[col_name] = scale_df[col_name].astype("object")
                scale_df[col_name] = scale_df[col_name].replace(to_replace=conversion_map)

        elif scale_name == "phx01":
            """
            Note: the raw files had these unique values per column (in order of the list below)
            [nan  0.  1.  2.]
            [nan  0.  2.  1.]
            [nan  0.  1.  2.]
            [nan  0.  2.  1.]
            [nan  0.  2.  1.]
            [0 1]
            
            After conversion, this is the result:
            [nan  1.  2.]
            [nan  2.  1.]
            [nan  1.  2.]
            [nan  2.  1.]
            [nan  2.  1.]
            [nan] <-- this is for bulimia. There were no actual values other than 0 or 1, so there is no one-hot encoding that will occur.
            We may need to manually create the columns for 2/5, 3, 4 and set them all to 0 (false).
            """
            cols_to_convert = ['alcoh', 'amphet', 'cannibis', 'opioid', 'ax_cocaine', 'bulimia']
            for col_name in cols_to_convert:
                if col_name == "bulimia":
                    conversion_map = {0: np.nan, 1: np.nan, 2: "2/5", 5: "2/5"}
                else:
                    conversion_map = {0: np.nan}
                scale_df[col_name] = scale_df[col_name].astype("object")
                scale_df[col_name] = scale_df[col_name].replace(to_replace=conversion_map)

            scale_df["bulimia||2/5"] = 0
            scale_df["bulimia||3"] = 0
            scale_df["bulimia||4"] = 0

        elif scale_name in ["ccv01_w2", "idsc01", "qids01_w0c"]:
            # No special conversion steps prior to one-hot encoding
            pass

        if cols_to_one_hot_encode is None or len(cols_to_one_hot_encode) == 0:
            scale_df = scale_df
        else:
            scale_df = one_hot_encode(scale_df, cols_to_one_hot_encode)
            scale_df = scale_df.drop(columns=cols_to_one_hot_encode)

        output_file_name = ONE_HOT_ENCODED_PREFIX + scale_name
        scale_df.to_csv(output_one_hot_encoded_dir_path + output_file_name + CSV_SUFFIX, index=False)

def convert_values(root_data_dir_path):
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_one_hot_encoded_dir_path = output_dir_path + "/" + DIR_ONE_HOT_ENCODED + "/"
    output_values_converted_dir_path = output_dir_path + "/" + DIR_VALUES_CONVERTED + "/"

    input_dir_path = output_one_hot_encoded_dir_path

    print("\n--------------------------------4. VALUE CONVERSION-----------------------------------\n")

    for filename in os.listdir(input_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_values_converted_dir_path):
            os.mkdir(output_values_converted_dir_path)

        if "rs__cs__ohe__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        print("*************************************************************")
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file
        scale_df = pd.read_csv(input_dir_path + "/" + filename, skiprows=[1])

        for col_name in scale_df.columns.values:
            for key, dict in VALUE_CONVERSION_MAP.items():
                if key == "minus":
                    continue
                elif col_name in dict["col_names"]:
                    scale_df[col_name] = scale_df[col_name].astype("object")
                    scale_df[col_name] = scale_df[col_name].replace(to_replace=dict["conversion_map"])
                elif key == "blank_to_zero":
                    if col_name in dict["col_names"]:
                        scale_df = handle_replace_if_row_null(scale_df, col_name)

        if scale_name == "sfhs01":
            config = VALUE_CONVERSION_MAP["minus"]
            for key, list_of_cols in config.items():
                print(key, list_of_cols)
                for col_name in list_of_cols:
                    conversion_map = {}
                    for k, value in scale_df[col_name].iteritems():
                        if value in conversion_map:
                            continue
                        elif key == 6 or key == 3:
                            conversion_map[value] = key - value
                        elif key == 1:
                            conversion_map[value] = value - 1
                    scale_df[col_name] = scale_df[col_name].replace(to_replace=conversion_map)
            
        output_file_name = VALUES_CONVERTED_PREFIX + scale_name
        scale_df.to_csv(output_values_converted_dir_path + output_file_name + CSV_SUFFIX, index=False)

def handle_replace_if_row_null(df, col_name):
    """
    Handles blank to zero conversion for rows which are null for a given scale. 
    """
    for i, row in df.iterrows():
        # If all column values are empty for this row, then leave it all null
        if sum(row.isnull()) == len(row):
            continue
        # But if there are non-empty values, then convert the col_name value to 0
        else:
            df.set_value(i, col_name, 0)
    return df

def aggregate_rows(root_data_dir_path):
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_values_converted_dir_path = output_dir_path + "/" + DIR_VALUES_CONVERTED + "/"
    output_aggregated_rows_dir_path = output_dir_path + "/" + DIR_AGGREGATED_ROWS + "/"

    input_dir_path = output_values_converted_dir_path

    print("\n--------------------------------5. ROW AGGREGATION-----------------------------------\n")

    main_keys = ['subjectkey', 'gender||F', 'gender||M', 'interview_age']
    aggregated_df = pd.DataFrame()

    for i, filename in enumerate(os.listdir(input_dir_path)):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_aggregated_rows_dir_path):
            os.mkdir(output_aggregated_rows_dir_path)

        if "rs__cs__ohe__vc__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        print("*************************************************************")
        print("Handling scale =", scale_name, ", filename =", filename)

        # Read in the txt file
        scale_df = pd.read_csv(input_dir_path + "/" + filename, skiprows=[1])

        # Append scale name and version to the column name
        cols = {}
        for col_name in scale_df.columns.values:
            if col_name in main_keys:
                continue
            else:
                cols[col_name] = scale_name + "__" + str(col_name)
        scale_df = scale_df.rename(columns = cols)

        if i == 0:
            aggregated_df = scale_df
        else:
            aggregated_df["subjectkey"] = aggregated_df["subjectkey"].astype(object)
            scale_df["subjectkey"] = scale_df["subjectkey"].astype(object)

            # The left df has to be the one with more rows, as joining the two will ensure all subjects are grabbed.
            if aggregated_df.shape[0] >= scale_df.shape[0]:
                left = aggregated_df
                right = scale_df
            else:
                left = scale_df
                right = aggregated_df

            aggregated_df = left.merge(right, on="subjectkey", how="left")

    output_file_name = AGGREGATED_ROWS_PREFIX + "stard_data_matrix"
    aggregated_df = aggregated_df.reindex(columns=(main_keys + list([a for a in aggregated_df.columns if a not in main_keys])))
    aggregated_df.to_csv(output_aggregated_rows_dir_path + output_file_name + CSV_SUFFIX, index=False)


def one_hot_encode(df, columns):
    # Convert categorical variables to indicator variables via one-hot encoding
    for col_name in columns:
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name, prefix_sep="||")], axis=1)
    return df

def drop_empty_columns(df):
    return df.dropna(axis="columns", how="all")  # Drop columns that are all empty

if __name__ == "__main__":
    data_dir_path = sys.argv[1]
    option = sys.argv[2]
    is_valid = len(sys.argv) == 3 and os.path.isdir(data_dir_path)

    if is_valid and option in ["--row-select", "-rs"]:
        select_rows(data_dir_path)

    elif is_valid and option in ["--column-select", "-cs"]:
        select_columns(data_dir_path)

    elif is_valid and option in ["--one-hot-encode", "-ohe"]:
        one_hot_encode_scales(data_dir_path)

    elif is_valid and option in ["--value-convert", "-vc"]:
        convert_values(data_dir_path)

    elif is_valid and option in ["--aggregate-rows", "-ag"]:
        aggregate_rows(data_dir_path)

    elif is_valid and option in ["--run-all", "-a"]:
        select_rows(data_dir_path)
        select_columns(data_dir_path)
        one_hot_encode_scales(data_dir_path)
        convert_values(data_dir_path)
        aggregate_rows(data_dir_path)

    else:
        raise Exception("Enter valid arguments\n"
              "\t path: the path to a real directory\n"
              "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3")
