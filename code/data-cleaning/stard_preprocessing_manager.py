import os
import sys
import pandas as pd
import numpy as np
from collections import namedtuple

from utils import *
from stard_preprocessing_globals import ORIGINAL_SCALE_NAMES, SCALES, VALUE_CONVERSION_MAP, \
    VALUE_CONVERSION_MAP_IMPUTE, NEW_FEATURES

""" 
This will take in multiple text files (representing psychiatric scales) and output multiple CSV files, at least for each scale read in.
"""

ROW_SELECTION_PREFIX = "rs__"
COLUMN_SELECTION_PREFIX = ROW_SELECTION_PREFIX + "cs__"
ONE_HOT_ENCODED_PREFIX = COLUMN_SELECTION_PREFIX + "ohe__"
VALUES_CONVERTED_PREFIX = ONE_HOT_ENCODED_PREFIX + "vc__"
AGGREGATED_ROWS_PREFIX = VALUES_CONVERTED_PREFIX + "ag__" # Final: "rs__cs__ohe__vc__ag__" which represents the order of the pipeline
IMPUTED_PREFIX = AGGREGATED_ROWS_PREFIX + "im__"
CSV_SUFFIX = ".csv"

DIR_PROCESSED_DATA = "processed_data"
DIR_ROW_SELECTED = "row_selected_scales"
DIR_COLUMN_SELECTED = "column_selected_scales"
DIR_ONE_HOT_ENCODED = "one_hot_encoded_scales"
DIR_VALUES_CONVERTED = "values_converted_scales"
DIR_AGGREGATED_ROWS = "aggregated_rows_scales"
DIR_IMPUTED = "imputed_scales"
DIR_Y_MATRIX = "y_matrix"

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
    """
    Handles converting values for different variables per scale. This is similar to the imputation step, in that certain
    values are being derived, however the difference is that this step handles it solely on the scale-level. For (1) generic
    value conversion that is common across all scales, or (2) value conversion/imputation is dependent on values of features
    between different scales, then this will be handled in step #6, imputation. 
    """

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

def impute(root_data_dir_path):
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_aggregated_rows_dir_path = output_dir_path + "/" + DIR_AGGREGATED_ROWS + "/"
    output_imputed_dir_path = output_dir_path + "/" + DIR_IMPUTED + "/"

    # The input directory path will be that from the previous step (#5), row aggregation.
    input_dir_path = output_aggregated_rows_dir_path

    print("\n--------------------------------6. IMPUTATION-----------------------------------\n")

    final_data_matrix = pd.DataFrame()

    for i, filename in enumerate(os.listdir(input_dir_path)):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_imputed_dir_path):
            os.mkdir(output_imputed_dir_path)

        if "rs__cs__ohe__vc__ag__" not in filename:
            continue

        scale_name = filename.split(".")[0].split("__")[-1]

        print("*************************************************************")
        print("Handling full data matrix =", scale_name, ", filename =", filename)

        # Read in the txt file
        agg_df = pd.read_csv(input_dir_path + "/" + filename, skiprows=[1])

        # Handle replace with mean or median
        agg_df = replace_with_median(agg_df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_median"]["col_names"]))
        agg_df = replace_with_mode(agg_df, list(VALUE_CONVERSION_MAP_IMPUTE["blank_to_mode"]["col_names"]))

        # Handle direct value conversions (NaN to a specific number)
        blank_to_zero_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_zero"]
        blank_to_one_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_one"]
        blank_to_twenty_config = VALUE_CONVERSION_MAP_IMPUTE["blank_to_twenty"]
        agg_df = replace(agg_df, list(blank_to_zero_config["col_names"]), blank_to_zero_config["conversion_map"])
        agg_df = replace(agg_df, list(blank_to_one_config["col_names"]), blank_to_one_config["conversion_map"])
        agg_df = replace(agg_df, list(blank_to_twenty_config["col_names"]), blank_to_twenty_config["conversion_map"])

        crs01_df = pd.read_csv(root_data_dir_path + "/crs01.txt", sep="\t", skiprows=[1])

        for new_feature in NEW_FEATURES:
            agg_df[new_feature] = np.nan

        # Handle imputation based on cross-column conditions
        for i, row in agg_df.iterrows():
            if ('gender||F' in row and 'gender||M' in row) and (np.isnan(row['gender||F']) or np.isnan(row['gender||M'])):
                    gender = crs01_df.loc[crs01_df['subjectkey'] == row['subjectkey']]['gender'].iloc[0]
                    if gender == "M":
                        agg_df.set_value(i, 'gender||F', 0)
                        agg_df.set_value(i, 'gender||M', 1)
                    elif gender == "F" or np.isnan(gender):
                        agg_df.set_value(i, 'gender||F', 1)
                        agg_df.set_value(i, 'gender||M', 0)
            if 'interview_age' in row and np.isnan(row['interview_age']):
                age = crs01_df.loc[crs01_df['subjectkey'] == row['subjectkey']]['interview_age'].iloc[0]
                if np.isnan(age):
                    print("Age is null", row["subjectkey"])
                    agg_df.set_value(i, 'interview_age', agg_df['interview_age'].median())
                else:
                    agg_df.set_value(i, 'interview_age', age)
            if 'ucq01__ucq010' in row:
                val = 1
                if row['ucq01__ucq010'] == 0:
                    val = 0
                agg_df.set_value(i, 'ucq01__ucq020', val)
                agg_df.set_value(i, 'ucq01__ucq030', val)
            if 'wpai01__wpai01' in row:
                if row['wpai01__wpai01'] == 1:
                    agg_df.set_value(i, 'dm01_w0__inc_curr', 1)
                    agg_df.set_value(i, 'dm01_w0__mempl', 2000)
                    agg_df.set_value(i, 'dm01_enroll__empl||1.0', 0)
                    agg_df.set_value(i, 'dm01_enroll__empl||3.0', 1)
                    agg_df.set_value(i, 'dm01_enroll__privins||0.0', 0)
                    agg_df.set_value(i, 'dm01_enroll__privins||1.0', 1)

                elif row['wpai01__wpai01'] == 0 or np.isnan(row['wpai01__wpai01']):
                    agg_df.set_value(i, 'dm01_w0__inc_curr', 0)
                    agg_df.set_value(i, 'dm01_w0__mempl', 0)
                    agg_df.set_value(i, 'dm01_enroll__empl||1.0', 1)
                    agg_df.set_value(i, 'dm01_enroll__privins||1.0', 0)

                else:
                    agg_df.set_value(i, 'dm01_enroll__empl||3.0', 0)
                    agg_df.set_value(i, 'dm01_enroll__privins||0.0', 1)
                    agg_df.set_value(i, 'dm01_enroll__privins||1.0', 0)
            if 'wsas01__totwsas' in row:
                col_names = ['wsas01__wsas01', 'wsas01__wsas03', 'wsas01__wsas04', 'wsas01__wsas05']
                agg_df.set_value(i, 'wsas01__totwsas', np.sum(row[col_names]))
            if 'hrsd01__hdtot_r' in row:
                col_names = ['hrsd01__hsoin',
                             'hrsd01__hmnin',
                             'hrsd01__hemin',
                             'hrsd01__hmdsd',
                             'hrsd01__hinsg',
                             'hrsd01__happt',
                             'hrsd01__hwl',
                             'hrsd01__hsanx',
                             'hrsd01__hhypc',
                             'hrsd01__hvwsf',
                             'hrsd01__hsuic',
                             'hrsd01__hintr',
                             'hrsd01__hengy',
                             'hrsd01__hslow',
                             'hrsd01__hagit',
                             'hrsd01__hsex',
                             'hrsd01__hdtot_r']
                agg_df.set_value(i, 'hrsd01__hdtot_r', np.sum(row[col_names]))

            agg_df = add_new_imputed_features(agg_df, row, i)

        # Drop columns
        agg_df = agg_df.drop(columns=['wsas01__wsastot'])

        # Loop through each row, and replace specific columns by the combination of other columns.
        final_data_matrix = agg_df

    output_file_name = IMPUTED_PREFIX + "stard_data_matrix"
    final_data_matrix.to_csv(output_imputed_dir_path + output_file_name + CSV_SUFFIX, index=False)
    print("File has been written to:", output_imputed_dir_path + output_file_name + CSV_SUFFIX)

def add_new_imputed_features(df, row, i):
    imput_anyanxiety = ['phx01__psd', 'phx01__pd_ag', 'phx01__pd_noag', 'phx01__specphob', 'phx01__soc_phob', 'phx01__gad_phx']
    val = 1 if sum(row[imput_anyanxiety] == 1) > 0 else 0
    df.set_value(i, 'imput_anyanxiety', val)

    imput_bech = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hpanx', 'hrsd01__heng']
    df.set_value(i, 'imput_bech', np.sum(row[imput_bech]))

    imput_maier = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hpanx', 'hrsd01__heng', 'hrsd01__hagit']
    df.set_value(i, 'imput_maier', np.sum(row[imput_maier]))

    imput_santen = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hpanx', 'hrsd01__heng', 'hrsd01__hsuic']
    df.set_value(i, 'imput_santen', np.sum(row[imput_santen]))

    imput_gibbons = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hpanx', 'hrsd01__heng', 'hrsd01__hsuic', 'hrsd01__hagit', 'hrsd01__hsanx', 'hrsd01__hsex']
    df.set_value(i, 'imput_gibbons', np.sum(row[imput_gibbons]))

    imput_hamd7 = ['hrsd01__hmdsd', 'hrsd01__hvwsf', 'hrsd01__hintr', 'hrsd01__hpanx', 'hrsd01__hsanx', 'hrsd01__ hengy', 'hrsd01__hsuicide']
    df.set_value(i, 'imput_hamd7', np.sum(row[imput_hamd7]))

    imput_hamdret = ['hrsd01__hmdsd', 'hrsd01__hintr', 'hrsd01__hslow', 'hrsd01__hsex']
    df.set_value(i, 'imput_hamdret', np.sum(row[imput_hamdret]))

    imput_hamdanx = ['hrsd01__hpanx', 'hrsd01__hsanx', 'hrsd01__happt', 'hrsd01__hengy', 'hrsd01__hhypc']
    df.set_value(i, 'imput_hamdanx', np.sum(row[imput_hamdanx]))

    imput_hamdsle = ['hrsd01__hsoin', 'hrsd01__hmnin', 'hrsd01__hemin']
    df.set_value(i, 'imput_hamdsle', np.sum(row[imput_hamdsle]))

    imput_idsc5w0 = ['qids01_w0c__vmdsd', 'qids01_w0c__vintr', 'qids01_w0c__vengy', 'qids01_w0c__vvwsf', 'qids01_w0c__vslow']
    val_imput_idsc5w0 = np.sum(row[imput_idsc5w0])
    df.set_value(i, 'imput_idsc5w0', val_imput_idsc5w0)

    imput_idsc5w2 = ['qids01_w2c__vmdsd', 'qids01_w2c__vintr', 'qids01_w2c__vengy', 'qids01_w2c__vvwsf', 'qids01_w2c__vslow']
    val_imput_idsc5w2 = np.sum(row[imput_idsc5w2])
    df.set_value(i, 'imput_idsc5w2', val_imput_idsc5w2)

    val = round((val_imput_idsc5w2 - val_imput_idsc5w0) / val_imput_idsc5w0 if val_imput_idsc5w0 else 0, 3)
    df.set_value(i, 'imput_idsc5pccg', val)

    val = round((row['qids01_w2c__qstot'] - row['qids01_w0c__qstot']) / row['qids01_w0c__qstot'] if row['qids01_w0c__qstot'] else 0, 3)
    df.set_value(i, 'imput_qidscpccg', val)

    return df

def replace_with_median(df, col_names):
    if set(col_names).issubset(df.columns):
        df[col_names] = df[col_names].apply(lambda col: col.fillna(col.median()), axis=0)
    return df

def replace_with_mode(df, col_names):
    if set(col_names).issubset(df.columns):
        df[col_names] = df[col_names].apply(lambda col: col.fillna(col.mode()), axis=0)
    return df

def replace(df, col_names, conversion_map):
    if set(col_names).issubset(df.columns):
        df[col_names] = df[col_names].replace(to_replace=conversion_map)
        print("Replaced", conversion_map)
    return df

def one_hot_encode(df, columns):
    # Convert categorical variables to indicator variables via one-hot encoding
    for col_name in columns:
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name, prefix_sep="||")], axis=1)
    return df

def drop_empty_columns(df):
    return df.dropna(axis="columns", how="all")  # Drop columns that are all empty

def generate_y(root_data_dir_path):
    output_dir_path = root_data_dir_path + "/" + DIR_PROCESSED_DATA
    output_y_dir_path = output_dir_path + "/" + DIR_Y_MATRIX + "/"

    print("\n--------------------------------7. Y MATRIX GENERATION-----------------------------------\n")

    y_lvl2_rem_ccv01 = pd.DataFrame()
    y_lvl2_rem_qids01 = pd.DataFrame()

    for filename in os.listdir(root_data_dir_path):
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        if not os.path.exists(output_y_dir_path):
            os.mkdir(output_y_dir_path)

        scale_name = filename.split(".")[0]
        if scale_name not in ['ccv01', 'qids01']:
            continue

        curr_scale_path = root_data_dir_path + "/" + filename

        # Read in the txt file + preliminary processing
        scale_df = pd.read_csv(curr_scale_path, sep='\t', skiprows=[1])

        print("*************************************************************")
        print("Handling scale = ", scale_name)

        if scale_name == "ccv01":
            scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("int")
            scale_df = scale_df.loc[scale_df['days_baseline'] > 21]

            i = 0
            for id, group in scale_df.groupby(['subjectkey']):
                y_lvl2_rem_ccv01.loc[i, "subjectkey"] = id
                subset = group[(group['level'] == "Level 1") | (group['level'] == "Level 2") & (group['remsn'] == 1)]
                if subset.shape[0] == 0:
                    y_lvl2_rem_ccv01.loc[i, "target"] = 0
                else:
                    y_lvl2_rem_ccv01.loc[i, "target"] = 1
                i += 1

        if scale_name == "qids01":
            # scale_df.loc[:, "days_baseline"] = scale_df["days_baseline"].astype("int")
            scale_df = scale_df.loc[scale_df['days_baseline'] > 21]

            i = 0
            for id, group in scale_df.groupby(['subjectkey']):
                y_lvl2_rem_qids01.loc[i, "subjectkey"] = id
                subset = group[(group['level'] == "Level 3") | (group['level'] == "Level 4")]
                if subset.shape[0] == 0:
                    y_lvl2_rem_qids01.loc[i, "target"] = 0
                else:
                    subset = group[(group['version_form'] == "Clinician") & (group['level'] != "Follow-Up") & (group['qstot'] <= 5)]
                    if subset.shape[0] > 0:
                        y_lvl2_rem_qids01.loc[i, "target"] = 1
                    else:
                        y_lvl2_rem_qids01.loc[i, "target"] = 0
                i += 1

    y_lvl2_rem_ccv01.to_csv(output_y_dir_path + "y_lvl2_rem_ccv01" + CSV_SUFFIX, index=False)
    y_lvl2_rem_qids01.to_csv(output_y_dir_path + "y_lvl2_rem_qids01" + CSV_SUFFIX, index=False)

    print("File has been written to:", output_y_dir_path + "y_lvl2_rem_ccv01" + CSV_SUFFIX)
    print("File has been written to:", output_y_dir_path + "y_lvl2_rem_qids01" + CSV_SUFFIX)


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

    elif is_valid and option in ["--impute", "-im"]:
        impute(data_dir_path)

    elif is_valid and option in ["--y-generation", "-y"]:
        generate_y(data_dir_path)

    elif is_valid and option in ["--run-all", "-a"]:
        select_rows(data_dir_path)
        select_columns(data_dir_path)
        one_hot_encode_scales(data_dir_path)
        convert_values(data_dir_path)
        aggregate_rows(data_dir_path)
        impute(data_dir_path)
        generate_y(data_dir_path)

    else:
        raise Exception("Enter valid arguments\n"
              "\t path: the path to a real directory\n"
              "\t e.g. python stard_preprocessing_manager.py /Users/teyden/Downloads/stardmarch19v3")
