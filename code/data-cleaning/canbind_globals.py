import numpy as np

"""
This is a globals file storing the configuration for managing the processing of CAN-BIND data. 

NOTES: ***********************
- All names of columns are standardized to uppercase.
- 
"""
# Collision manager
COLLISION_MANAGER = {}

# Names of columns to filter in or out
# TODO Note: for validation of data integrity, remove SUBJLABEL, EVENTNAME and VISITSTATUS to check against original CSVs
COL_NAMES_BLACKLIST_COMMON = ["EVENTNAME", "VISITSTATUS", "COHORT_ID", "GROUP", "VISITDATE",
                              "DOSAGE_ID", "RESPOND_WK16", "CHANGE_WK8", "CHANGE_WK16", "REMISSION_WK8", 
                              "REMISSION_WK16", "ARM"]
COL_NAMES_BLACKLIST_IPAQ = ["IPAQ_1", "IPAQ_2", "IPAQ_2_HOUR", "IPAQ_2_MIN", "IPAQ_3", "IPAQ_4", "IPAQ_4_HOUR",
                            "IPAQ_4_MIN", "IPAQ_5", "IPAQ_6", "IPAQ_6_HOUR", "IPAQ_6_MIN", "IPAQ_7", "IPAQ_7_HOUR",
                            "IPAQ_7_MIN", "VIG_TOTAL_TR", "MODERATE_TOTAL_TR", "WALKING_TOTAL_TR"]
COL_NAMES_BLACKLIST_QIDS = ["QIDS_RESP_WK8_baseline","QIDS_RESP_WK16_baseline","QIDS_RESP_WK16_week 2"]
COL_NAMES_BLACKLIST_LEAPS = ["LAM_1", "LAM_1_CLASSIFICATION_NUM1", "LAM_1_CLASSIFICATION_LABEL1",
                             "LAM_1_CLASSIFICATION_NUM2", "LAM_1_CLASSIFICATION_LABEL2"]
COL_NAMES_BLACKLIST_MINI = []
COL_NAMES_WHITELIST_PSYHIS = {"PSYHIS_FH": True, "PSYHIS_MDD_AGE": True, "PSYHIS_MDE_NUM": True,
                              "PSYHIS_MDE_EP_DUR_MO": True, "PSYHIS_MDE_CUR_TX": True, "PSYHIS_MDD_HIST": True,
                              "PSYHIS_MDD_PREV": True}
COL_NAMES_BLACKLIST_PSYHIS = []
COL_NAMES_BLACKLIST_DEMO = ["ETHNCTY_CDN___9998", "ETHNCTY_CDN_OTH", "JOB_CLASS"]
# COL_NAMES_BLACKLIST_DARS = ["DARS_A_1", "DARS_A_2", "DARS_C_1", "DARS_C_2", "DARS_C_3", "DARS_E_1", "DARS_E_2",
#                             "DARS_G_1", "DARS_G_2", "DARS_A_3", "DARS_E_3", "DARS_G_3"]
COL_NAMES_BLACKLIST_DARS = []
COL_NAMES_BLACKLIST_SHAPS = []
COL_NAMES_BLACKLIST_UNIQS = []
COL_NAMES_BLACKLIST_YGEN = []

# Names of columns to one-hot encode
COL_NAMES_ONE_HOT_ENCODE = ["SITESYMBOL", "SEX", "MVPA_GROUP_screening", "VIG_GROUP_screening", "MINI_BP_I_TIME",
                            "MINI_BP_II_TIME", "MINI_HME_TIME", "MINI_BP_NOS_TIME", "MINI_ME_TIME",
                            "MINI_PSYCTC_DSRDR_TIME", "MINI_MOOD_DSRDR_PSYCTC_TIME", "MINI_SOCL_PHOBIA_SUBTYPE",
                            "MINI_PRMRY_DX", "EMPLOY_STATUS", "HANDEDNESS", "MRTL_STATUS", "LAM_0_baseline"]

# Names of columns with NA values
COL_NAMES_NA = []

# Column key names
COL_NAME_GROUP = "GROUP"
COL_NAME_EVENTNAME = "EVENTNAME"
COL_NAME_PATIENT_ID = "SUBJLABEL"
COL_NAME_VISITSTATUS = "VISITSTATUS"

# Names of whitelist values for the columns EVENTNAME and VISITSTATUS (IPAQ only), for keeping just the entries within week 2
EVENTNAME_WHITELIST = ["screening", "baseline", "week 2"]
YGEN_EVENTNAME_WHITELIST = ["screening", "baseline", "week 2","week 4", "week 8"]
VISITSTATUS_WHITELIST = ["screening/baseline"]

# Names of whitelist values for the column GROUP
GROUP_WHITELIST = ["control"]

#### LIST MANGERS FOR EXTENSION OF COLUMNS BASED ON EVENT
"""
New columns are generated for these following columns, where a column is named ABC and we care about
baseline and week 2 events for which ABC values are recorded, then columns ABC_baseline and ABC_week 2 will be created.
"""

## NOTE: The ones with empty lists are generated computationally because there are no exclusions of columns

COL_NAMES_NEW_FROM_EXTENSION = []
COL_NAMES_TO_DROP_FROM_EXTENSION = []

# Names of MADRS columns to convert
EVENTNAME_WHITELIST_MADRS = ["baseline", "week 2"]
COL_NAMES_MADRS_TO_CONVERT = [] # Include all

# Names of QIDS columns to convert
EVENTNAME_WHITELIST_QIDS = ["baseline", "week 2"]
COL_NAMES_QIDS_TO_CONVERT = [] # Include all

# Names of QIDS columns to convert for y geneation
YGEN_EVENTNAME_WHITELIST_QIDS = ["baseline", "week 2", "week 4", "week 8"]
COL_NAMES_QIDS_TO_CONVERT = [] # Include all

# Names of GAD-7 columns to convert
EVENTNAME_WHITELIST_GAD7 = ["baseline", "week 2"]
COL_NAMES_GAD7_TO_CONVERT = [] # Include all

# Names of HCL columns to convert
EVENTNAME_WHITELIST_HCL = ["screening"]
COL_NAMES_HCL_TO_CONVERT = [] # Include all

# Names of IPAQ columns to convert
EVENTNAME_WHITELIST_IPAQ = ["screening"]
COL_NAMES_IPAQ_TO_CONVERT = ["VIG_TOTAL", "MOD_TOTAL", "WALKING_TOTAL", "SITTING_AVERAGE", "MVPA_GROUP", "VIG_GROUP", "MVPA"]

# Names of LEAPS columns to convert
EVENTNAME_WHITELIST_LEAPS = ["baseline"]
COL_NAMES_LEAPS_TO_CONVERT = ["LAM_0", "LAM_2", "LAM_3", "LAM_4_A", "LAM_4_B", "LAM_4_C", "LAM_4_D", "LAM_4_E",
                              "LAM_4_F", "LAM_4_G", "LAM_TOT_OVERALL", "LAM_TOT_PRODUCT", "LAM_TOT_ABSENT"]

# Names of LEAPS columns to convert
EVENTNAME_WHITELIST_SDS = ["baseline"]
COL_NAMES_SDS_TO_CONVERT = ["SDS_1_1", "SDS_2_1", "SDS_3_1", "SDS_4", "SDS_TOT", "SDS_FUNC_RESP", "SDS_FUNC_REMISS"]

# Names of QLESQ columns to convert
EVENTNAME_WHITELIST_QLESQ = ["baseline"]
COL_NAMES_QLESQ_TO_CONVERT = [] # Include all

# Names of DARS columns to convert
EVENTNAME_WHITELIST_DARS = ["baseline"]
COL_NAMES_DARS_TO_CONVERT = ["DARS_B_1", "DARS_B_2", "DARS_B_3", "DARS_B_4", "DARS_B_5", "DARS_B_6", "DARS_B_7",
                             "DARS_B_8", "DARS_B_9", "DARS_D_10", "DARS_D_11", "DARS_D_12", "DARS_D_13", "DARS_D_14",
                             "DARS_D_15", "DARS_F_16", "DARS_F_17", "DARS_F_18", "DARS_F_19", "DARS_F_20", "DARS_F_21",
                             "DARS_H_22", "DARS_H_23", "DARS_H_24", "DARS_H_25", "DARS_H_26"] # Include all numerical
NEW_COL_NAMES_DARS = []

# Names of SHAPS columns to convert
EVENTNAME_WHITELIST_SHAPS = ["baseline"]
COL_NAMES_SHAPS_TO_CONVERT = [] # Include all
NEW_COL_NAMES_SHAPS = []

# Scales where the event information is stored in the VISITSTATUS column instead
ALTERNATIVE_EVENT_COLUMNS = ["HCL", "IPAQ"]

COL_NAMES_TO_CONVERT = [("MADRS", EVENTNAME_WHITELIST_MADRS, COL_NAMES_MADRS_TO_CONVERT),
                        ("QIDS", EVENTNAME_WHITELIST_QIDS, COL_NAMES_QIDS_TO_CONVERT),
                        ("GAD7", EVENTNAME_WHITELIST_GAD7, COL_NAMES_GAD7_TO_CONVERT),
                        ("IPAQ", EVENTNAME_WHITELIST_IPAQ, COL_NAMES_IPAQ_TO_CONVERT),
                        ("LEAPS", EVENTNAME_WHITELIST_LEAPS, COL_NAMES_LEAPS_TO_CONVERT),
                        ("HCL", EVENTNAME_WHITELIST_HCL, COL_NAMES_HCL_TO_CONVERT),
                        ("SDS", EVENTNAME_WHITELIST_SDS, COL_NAMES_SDS_TO_CONVERT),
                        ("QLESQ", EVENTNAME_WHITELIST_QLESQ, COL_NAMES_QLESQ_TO_CONVERT)]

YGEN_COL_NAMES_TO_CONVERT = [("QIDS", YGEN_EVENTNAME_WHITELIST_QIDS, COL_NAMES_QIDS_TO_CONVERT)]


#### LIST AND MAP MANAGERS FOR COLUMNS WITH DATA TO BE REPLACED

SEX_VALUES_MAP = {
    "col_names": ["SEX"],
    "values": {
        1: "female",
        2: "male",
        9999: np.nan,
        4: np.nan
    }
}
MINI_LIFETIME_CURRENT_MAP = {
    "col_names": ["MINI_PSYCTC_DSRDR_TIME", "MINI_MOOD_DSRDR_PSYCTC_TIME"],
    "values": {
        1: "current",
        2: "lifetime"
    }
}
MINI_PAST_CURRENT_MAP = {
    "col_names": ["MINI_BP_I_TIME", "MINI_BP_II_TIME", "MINI_HME_TIME", "MINI_BP_NOS_TIME", "MINI_ME_TIME"],
    "values": {
        1: "current",
        2: "past"
    }
}
MINI_GENERALIZED_MAP = {
    "col_names": ["MINI_SOCL_PHOBIA_SUBTYPE"],
    "values": {
        1: "generalized_current",
        2: "nongeneralized_current"
    }
}
# TODO consider np.nan for value 2, which will be converted to a median value
MINI_RULEDOUT_MAP = {
    "col_names": ["MINI_MED_RULED_OUT"],
    "values": {
        0: 0,
        1: 1,
        2: 0
    }
}
YN_MAP = {
    "col_names": ["QLESQ_0_baseline"],
    "values": {
        "N": 0,
        "Y": 1
    }
}
YESNO_MAP = {
    "col_names": ["SDS_FUNC_RESP_baseline", "SDS_FUNC_REMISS_baseline"],
    "values": {
        "No": 0,
        "Yes": 1
    }
}
IGNORE_MAP = {
    "col_names": ["HANDEDNESS", "EMPLOY_STATUS"],
    "values": {
        9999: np.nan,
        9998: np.nan,
        9996: np.nan,
    }
}
HANDEDNESS_MAP = {
    "col_names": ["HANDEDNESS"],
    "values": {
        1: "Left",
        2: "Right",
        3: "Ambidextrous",
    }
}
MRTL_STATUS_MAP = {
    "col_names": ["MRTL_STATUS"],
    "values": {
        1: "Never Married",
        2: "Separated",
        3: "Married",
        4: "Divorced",
        5: "Domestic Partnership",
        6: "Widowed"
    }
}
# Note: there are more possibilities for the one below but I only added the ones actually seen in the data
MINI_PRMRY_DX_MAP = {
    "col_names": ["MINI_PRMRY_DX"],
    "values": {
        1: "MAJOR DEPRESSIVE DISORDER; Current (2 weeks)",
        2: "MAJOR DEPRESSIVE DISORDER; Past",
        3: "MAJOR DEPRESSIVE DISORDER; Recurrent"
    }
}
SDS_MAP = {
    "col_names": ["SDS_5", "SDS_4_baseline"],
    "values": {
        "placeholder": np.nan,
        9998: 0,
    }
}
NA_TO_BLANK_MAP = {
    "col_names": ["SDS_1_1_baseline", "QLESQ_1A_15_baseline"],
    "values": {
        "NA": 0,
    }
}
EDUC_MAP = {
    "col_names": ["EDUC"],
    "values": {
        14: 12, 15: 12, 16: 13, 17: 14, 18: 14, 19: 16, 20: 18, 21: 20, 22: 22,
        "NA": np.nan,
        "": np.nan
    }
}
TARGET_MAP = {
    "col_names": ["QIDS_RESP_WK8_week 8"],
    "values": {
        "NonResponder": 0,
        "Responder": 1,
    }
}
# BLANK_TO_ZERO_MAP = {
#     "col_names": ["MINI_PD_TIME_CUR", "MINI_PD_TIME_LIFE", "MINI_MDE_TIME_CUR", "MINI_MDE_TIME_PAST",
#                   "MINI_MDE_TIME_RECUR", "MINI_SUICDLTY_SCORE_RANGE", "VIG_TOTAL_screening", "MOD_TOTAL_screening",
#                   "WALKING_TOTAL_screening", "SITTING_AVERAGE_screening","PSYHIS_MDD_PREV", "MVPA_screening"],
#     "values": {
#         "": 0
#     }
# }
VALUE_REPLACEMENT_MAPS = [SEX_VALUES_MAP, MINI_LIFETIME_CURRENT_MAP, MINI_PAST_CURRENT_MAP, MINI_GENERALIZED_MAP,
                          MINI_RULEDOUT_MAP, YN_MAP, YESNO_MAP, IGNORE_MAP, NA_TO_BLANK_MAP, SDS_MAP, EDUC_MAP,
                          HANDEDNESS_MAP, MRTL_STATUS_MAP, MINI_PRMRY_DX_MAP]#, BLANK_TO_ZERO_MAP]


####
UNIQ_COLUMNS = {}
COL_NAMES_CATEGORICAL = {}
COL_NAMES_NA = {}

FILENAMES = []

NUM_DATA_FILES = 0
NUM_DATA_ROWS = 0
NUM_DATA_COLUMNS = 0

QLESQ_COL_MAPPING = {
    "QLESQ_1A_1_baseline": "QLESQ_1B_1_baseline",
    "QLESQ_1A_2_baseline": "QLESQ_1B_2_baseline",
    "QLESQ_1A_3_baseline": "QLESQ_1B_3_baseline",
    "QLESQ_1A_4_baseline": "QLESQ_1B_4_baseline",
    "QLESQ_1A_5_baseline": "QLESQ_1B_5_baseline",
    "QLESQ_1A_6_baseline": "QLESQ_1B_6_baseline",
    "QLESQ_1A_7_baseline": "QLESQ_1B_7_baseline",
    "QLESQ_1A_8_baseline": "QLESQ_1B_8_baseline",
    "QLESQ_1A_9_baseline": "QLESQ_1B_9_baseline",
    "QLESQ_1A_10_baseline": "QLESQ_1B_10_baseline",
    "QLESQ_1A_11_baseline": "QLESQ_1B_11_baseline",
    "QLESQ_1A_12_baseline": "QLESQ_1B_12_baseline",
    "QLESQ_1A_13_baseline": "QLESQ_1B_13_baseline",
    "QLESQ_1A_14_baseline": "QLESQ_1B_14_baseline",
    "QLESQ_1A_16_baseline": "QLESQ_1B_16_baseline",
    "QLESQA_TOT_baseline": "QLESQB_TOT_baseline"
}

# JJ added following for the imputation step
VALUE_CONVERSION_MAP_IMPUTE = {
    "blank_to_zero": {
        "col_names": {'LAM_2_baseline','LAM_3_baseline','LAM_4_A_baseline','LAM_4_B_baseline','LAM_4_C_baseline',
                      'LAM_4_D_baseline','LAM_4_E_baseline','LAM_4_F_baseline','LAM_4_G_baseline','LAM_TOT_OVERALL_baseline',
                      'LAM_TOT_PRODUCT_baseline','LAM_TOT_ABSENT_baseline','SDS_1_1_baseline','QLESQ_1A_15_baseline',
                      "PSYHIS_MDD_PREV","MINI_PD_TIME_CUR", "MINI_PD_TIME_LIFE", "MINI_MDE_TIME_CUR", "MINI_MDE_TIME_PAST",
                  "MINI_MDE_TIME_RECUR", "MINI_SUICDLTY_SCORE_RANGE", "VIG_TOTAL_screening", "MOD_TOTAL_screening",
                  "WALKING_TOTAL_screening", "SITTING_AVERAGE_screening","PSYHIS_MDD_PREV", "MVPA_screening"},
        "conversion_map": {np.nan: 0}
    },
    "blank_to_one": {
        "col_names": {},
        "conversion_map": None
    },
    "blank_to_twenty": {
        "col_names": {},
        "conversion_map": None
    },
    "blank_to_median": {
        "col_names": {'EDUC','MINI_AGRPHOBIA_TIME','MINI_APD_TIME','MINI_MED_RULED_OUT','PSYHIS_FH','PSYHIS_MDD_AGE','PSYHIS_MDE_EP_DUR_MO','PSYHIS_MDE_CUR_TX','PSYHIS_MDE_NUM','MADRS_APRNT_SDNS_baseline','MADRS_APRNT_SDNS_week2','MADRS_CONC_DFCTY_baseline','MADRS_CONC_DFCTY_week2','MADRS_INBLTY_TO_FEEL_baseline','MADRS_INBLTY_TO_FEEL_week2','MADRS_INN_TNSN_baseline','MADRS_INN_TNSN_week2','MADRS_LASS_baseline','MADRS_LASS_week2','MADRS_PESS_THTS_baseline','MADRS_PESS_THTS_week2','MADRS_RDCD_APTIT_baseline','MADRS_RDCD_APTIT_week2','MADRS_RDCD_SLP_baseline','MADRS_RDCD_SLP_week2','MADRS_RPTRD_SDNS_baseline','MADRS_RPTRD_SDNS_week2','MADRS_SUICDL_THTS_baseline','MADRS_SUICDL_THTS_week2','GAD7_1_1_week2','GAD7_1_2_baseline','GAD7_1_2_week2','GAD7_1_3_baseline','GAD7_1_3_week2','GAD7_1_4_baseline','GAD7_1_4_week2','GAD7_1_5_baseline','GAD7_1_5_week2','GAD7_1_6_baseline','GAD7_1_6_week2','GAD7_1_7_baseline','GAD7_1_7_week2','GAD7_2_baseline','GAD7_2_week2','GAD7_TOT_baseline','GAD7_TOT_week2','HCL_1_screening','HCL_2_screening','HCL_3_0_screening','HCL_3_1_screening','HCL_3_2_screening','HCL_3_3_screening','HCL_3_4_screening','HCL_3_5_screening','HCL_3_6_screening','HCL_3_7_screening','HCL_3_8_screening','HCL_3_9_screening','HCL_3_10_screening','HCL_3_11_screening','HCL_3_12_screening','HCL_3_13_screening','HCL_3_14_screening','HCL_3_15_screening','HCL_3_16_screening','HCL_3_17_screening','HCL_3_18_screening','HCL_3_19_screening','HCL_3_20_screening','HCL_3_21_screening','HCL_3_22_screening','HCL_3_23_screening','HCL_3_24_screening','HCL_3_25_screening','HCL_3_26_screening','HCL_3_27_screening','HCL_3_28_screening','HCL_3_29_screening','HCL_3_30_screening','HCL_3_31_screening','HCL_3_32_screening','HCL_TOTAL_screening','SDS_2_1_baseline','SDS_3_1_baseline','SDS_4_baseline','QIDS_SR_3_baseline','QIDS_SR_6_baseline','QIDS_SR_8_baseline','QIDS_SR_7_week2','QIDS_SR_9_week2','BL_HEIGHT','BL_WEIGHT','BL_BMI','SDS_5', 'QLESQ_1A_3_baseline_QLESQ_1B_3_baseline_merged'},
        "conversion_map": None
    },
    "blank_to_mode": {
        "col_names": {},
        "conversion_map": None
    },
    "qids_w0c_to_w0sr": {
        "col_names": {},
        "conversion_map": None
    },
    "qids_w0sr_to_w0c": {
        "col_names": {},
        "conversion_map": None
    },
    "qids_w2c_to_w2sr": {
        "col_names": {},
        "conversion_map": None
    }
}

# JJ added this for new feature imputation
NEW_FEATURES = []

# Store actual filenames for data, as version of data now includes multiple copies and the like
ORIGINAL_SCALE_FILENAMES = {
    "data_CBN01_1_X03_CLIN_BMI_0001.csv",
    "data_CBN01_DEMO_DATA_0002.csv",
    "data_CBN01_GAD7_DATA_0001.csv",
    "data_CBN01_HCL_DATA_121517.csv",
    "CBN_IPAQ_TRUNCATION_180531.xlsx",
    "data_CBN01_1_X03_CLIN_LEAPS_DATA_0001_REVIEW - VE copy.csv",
    "data_CBN01_1_X03_CLIN_MADRS_DATA_0002.csv",
    "data_CBN01_MINI_DATA_0002.csv",
    "data_CBN01_PSYHIS_DATA_0002.csv",
    "CBN01_QIDS_DATA_0002_20190911.csv",
    "data_CBN01_QLESQ_DATA_forREVEIW.csv",
    "data_CBN01_1_X03_CLIN_SDS_DATA_0001_REVIEW copy.csv",
    "data_CBN01_1_A01_V01_CSV_SHAPS_DARS_DEMOG.csv"
    }
    
YGEN_SCALE_FILENAMES = {"CBN01_QIDS_DATA_0002_20190911.csv"}

