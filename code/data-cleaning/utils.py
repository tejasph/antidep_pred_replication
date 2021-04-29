import numpy as np
import stard_globals as stard_gls

"""
Quick helper functions
"""


def is_empty_value(val):
    if val is None:
        return True
    elif val is np.nan or val != val:
        return True
    elif val == float('nan'):
        return True
    elif val == "" or val == "NA":
        return True
    return False

def get_valid_subjects(df):
    df = df.loc[(df.RESPOND_WK8.str.lower().isin(["responder", "nonresponder"]))]
    return df

def eliminate_early_leavers(orig_df):
    # Eliminate subjects that don't have any records > 21
    df = orig_df.copy(deep=True)
    df.days_baseline = df.days_baseline.astype("float")  # Deathly necessary. lol
    subjects_grouped = df.groupby([stard_gls.COL_NAME_SUBJECTKEY])

    subject_ids = []
    count = 0
    print("Number of subjects to begin with:", len(df["subjectkey"].unique()))
    for subject_id, group in subjects_grouped:
        for i, row in group.iterrows():
            # If a subject has at least one record > 21, then keep them
            if row.days_baseline > 21:
                subject_ids.append(subject_id)
                count += 1
                break

    reduced_df = df.loc[df.subjectkey.isin(subject_ids)]
    print("Number of subjects reduced to:", len(reduced_df["subjectkey"].unique()))
    return reduced_df
