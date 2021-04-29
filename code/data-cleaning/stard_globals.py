import numpy as np

"""
Globals used for STAR*D preprocessing and other processing
"""


ignr = np.nan

# Values in "level", "week", "days_baseline" are ordered in terms of optimal preference, as
# they should be used to filter rows of subjects based on this.
# Preference for:
#   smaller days_baseline values
#   larger week values
#   earlier level values
SCALES = {
    "dm01": {
        "level": ["Enrollment", "Level 1", "Level 2", "Level 3"],  # pref for lower/earlier
        "week": None,
        "days_baseline": None,
        "whitelist": ['resid', 'rtown', 'resy', 'resm', 'marital', 'spous', 'relat', 'frend', 'thous',
                      'educat', 'student', 'empl', 'volun', 'leave', 'publica', 'medicaid', 'privins',
                      'mkedc', 'enjoy', 'famim', 'interview_age', 'gender', 'subjectkey'],
        "validation_params": {
            "num_cols_start": 26,
            "num_cols_end": 26
        }
    },
    "dm01_alt_partial": {
        "level": ["Level 1"],
        "week": [0, 99],
        "days_baseline": None,
        "whitelist": ['mempl', 'assist', 'massist', 'unempl', 'munempl', 'otherinc', 'minc_other',
                      'totincom'],
        "validation_params": {
            "num_cols_start": 8,
            "num_cols_end": 8
        }
    },
    "ccv01": {
        "level": ["Level 1"],
        "week": [0.1, 2],
        "days_baseline": None,
        "whitelist": ['medication1_dosage', 'suicd', 'remsn', 'raise', 'effct', 'cncn', 'prtcl', 'stmed', 'trtmt'],
        "col_extenders": ['_Level 1_0.1', '_Level 1_2'],
        "validation_params": {
            "num_cols_start": 9,
            "num_cols_end": 18
        }
    },
    "crs01": {
        "level": None,
        "week": None,
        "days_baseline": None,
        "whitelist": ['heart', 'vsclr', 'hema', 'eyes', 'ugi', 'lgi', 'renal', 'genur', 'mskl', 'neuro', 'psych',
                      'respiratory', 'liverd', 'endod'],
        "validation_params": {
            "num_cols_start": 14,
            "num_cols_end": 14
        }
    },
    "hrsd01": {
        "level": ["Enrollment"],
        "week": None,
        "days_baseline": None,
        "whitelist": ['hsoin', 'hmnin', 'hemin', 'hmdsd', 'hpanx', 'hinsg', 'happt', 'hwl', 'hsanx', 'hhypc', 'hvwsf',
                      'hsuic', 'hintr', 'hengy', 'hslow', 'hagit', 'hsex', 'hdtot_r'],
        "validation_params": {
            "num_cols_start": 18,
            "num_cols_end": 18
        }
    },
    "mhx01": {
        "level": None,
        "week": None,
        "days_baseline": None,
        "whitelist": ['psmed'],
        "validation_params": {
            "num_cols_start": 1,
            "num_cols_end": 1
        }
    },
    "pdsq01": {
        "level": None,
        "week": None,
        "days_baseline": None,
        "whitelist": ['evy2w', 'joy2w', 'int2w', 'lap2w', 'gap2w', 'lsl2w', 'msl2w', 'jmp2w', 'trd2w', 'glt2w', 'neg2w',
                      'flr2w', 'cnt2w', 'dcn2w', 'psv2w', 'wsh2w', 'btr2w', 'tht2w', 'ser2w', 'spf2w', 'sad2y', 'apt2y',
                      'slp2y', 'trd2y', 'cd2y', 'low2y', 'hpl2y', 'trexp', 'trwit', 'tetht', 'teups', 'temem', 'tedis',
                      'teblk', 'termd', 'tefsh', 'teshk', 'tedst', 'tenmb', 'tegug', 'tegrd', 'tejmp', 'ebnge', 'ebcrl',
                      'ebfl', 'ebhgy', 'ebaln', 'ebdsg', 'ebups', 'ebdt', 'ebvmt', 'ebwgh', 'obgrm', 'obfgt', 'obvlt',
                      'obstp', 'obint', 'obcln', 'obrpt', 'obcnt', 'anhrt', 'anbrt', 'anshk', 'anrsn', 'anczy', 'ansym',
                      'anwor', 'anavd', 'pechr', 'pecnf', 'peslp', 'petlk', 'pevth', 'peimp', 'imagn', 'imspy', 'imdgr',
                      'impwr', 'imcrl', 'imvcs', 'fravd', 'frfar', 'frcwd', 'frlne', 'frbrg', 'frbus', 'frcar', 'fralo',
                      'fropn', 'franx', 'frsit', 'emwry', 'emstu', 'ematn', 'emsoc', 'emavd', 'emspk', 'emeat', 'emupr',
                      'emwrt', 'emstp', 'emqst', 'embmt', 'empty', 'emanx', 'emsit', 'dkmch', 'dkfam', 'dkfrd', 'dkcut',
                      'dkpbm', 'dkmge', 'dgmch', 'dgfam', 'dgfrd', 'dgcut', 'dgpbm', 'dgmge', 'wynrv', 'wybad', 'wysdt',
                      'wydly', 'wyrst', 'wyslp', 'wytsn', 'wycnt', 'wysnp', 'wycrl', 'phstm', 'phach', 'phsck', 'phpr',
                      'phcse', 'wiser', 'wistp', 'wiill', 'wintr', 'widr'],
        "validation_params": {
            "num_cols_start": 138,
            "num_cols_end": 138
        }
    },
    "phx01": {
        "level": None,
        "week": None,
        "days_baseline": None,
        "whitelist": ['dage', 'epino', 'episode_date', 'ai_none', 'alcoh', 'amphet', 'cannibis', 'opioid', 'pd_ag',
                      'pd_noag', 'specphob', 'soc_phob', 'ocd_phx', 'psd', 'gad_phx', 'axi_oth', 'aii_none', 'aii_def',
                      'aii_na', 'pd_border', 'pd_depend', 'pd_antis', 'pd_paran', 'pd_nos', 'axii_oth', 'dep', 'deppar',
                      'depsib', 'depchld', 'bip', 'bippar', 'bipsib', 'bipchld', 'alcohol', 'alcpar', 'alcsib',
                      'alcchld', 'drug_phx', 'drgpar', 'drgsib', 'drgchld', 'suic_phx', 'suicpar', 'suicsib',
                      'suicchld', 'wrsms', 'anorexia', 'bulimia', 'ax_cocaine'],
        "validation_params": {
            "num_cols_start": 49,
            "num_cols_end": 49
        }
    },
    "qlesq01": {
        "level": ["Level 1"],
        "week": None,
        "days_baseline": [0, 1, 2, 3, 4, 5, 6, 7],  # or less
        "whitelist": ['qlesq01', 'qlesq02', 'qlesq03', 'qlesq04', 'qlesq05', 'qlesq06', 'qlesq07', 'qlesq08', 'qlesq09',
                      'qlesq10', 'qlesq11', 'qlesq12', 'qlesq13', 'qlesq14', 'qlesq15', 'qlesq16', 'totqlesq'],
        "validation_params": {
            "num_cols_start": 17,
            "num_cols_end": 17
        }
    },
    "sfhs01": {
        "level": ["Level 1"],
        "week": None,
        "days_baseline": [0, 1, 2, 3, 4, 5, 6, 7],  # or less
        "whitelist": ['sfhs01', 'sfhs02', 'sfhs03', 'sfhs04', 'sfhs05', 'sfhs06', 'sfhs07', 'sfhs08', 'sfhs09',
                      'sfhs10', 'sfhs11', 'sfhs12', 'pcs12', 'mcs12'],
        "validation_params": {
            "num_cols_start": 14,
            "num_cols_end": 14
        }
    },
    "side_effects01": {
        "level": [1], # Numerical representation for this scale, sigh.
        "week": [2, 1, 0],
        "days_baseline": None,
        "whitelist": ['fisfq', 'fisin', 'grseb'],
        "validation_params": {
            "num_cols_start": 3,
            "num_cols_end": 3
        }
    },
    "ucq01": {
        "level": ["Level 1"],
        "week": None,
        "days_baseline": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # or less
        "whitelist": ['ucq010', 'ucq020', 'ucq030', 'ucq080', 'ucq091', 'ucq092', 'ucq100', 'ucq110', 'ucq120',
                      'ucq130', 'ucq140', 'ucq150', 'ucq160', 'ucq170', 'ucq040', 'ucq050', 'ucq060', 'ucq070'],
        "validation_params": {
            "num_cols_start": 20,
            "num_cols_end": 20
        }
    },
    "wpai01": {
        "level": ["Level 1"],
        "week": None,
        "days_baseline": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # or less
        "whitelist": ['wpai01', 'wpai02', 'wpai03', 'wpai04', 'wpai05', 'wpai06', 'wpai_totalhrs', 'wpai_pctmissed',
                      'wpai_pctworked', 'wpai_pctwrkimp', 'wpai_pctactimp', 'wpai_totwrkimp'],
        "validation_params": {
            "num_cols_start": 12,
            "num_cols_end": 12
        }
    },
    "wsas01": {
        "level": ["Level 1"],
        "week": None,
        "days_baseline": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # or less
        "whitelist": ['wsas01', 'wsas02', 'wsas03', 'wsas04', 'wsas05', 'totwsas', 'wsastot'],
        "validation_params": {
            "num_cols_start": 7,
            "num_cols_end": 7
        }
    },
    "prise01": {
        "level": ["Level 1"],
        "week": [2, 1, 0],
        "days_baseline": None,  # or less
        "whitelist": ['gdiar', 'gcnst', 'gdmth', 'gnone', 'gnsea', 'gstro', 'htplp', 'htdzy', 'htchs', 'htnone',
                      'heart_prs', 'skrsh', 'skpsp', 'skich', 'sknone', 'skdry', 'nvhed', 'nvtrm', 'nvcrd', 'nvnone',
                      'nvdzy', 'nrvsy', 'eyvsn', 'earng', 'enone', 'eyear', 'urdif', 'urpn', 'urmns', 'urfrq', 'urnone',
                      'genur_prs', 'sldif', 'slnone', 'slmch', 'sleep', 'sxls', 'sxorg', 'sxerc', 'sxnone', 'sex_prs',
                      'oaxty', 'octrt', 'omal', 'orsls', 'oftge', 'odegy', 'onone', 'other_prs', 'skin_c'],
        "validation_params": {
            "num_cols_start": 50,
            "num_cols_end": 50
        }
    },
    "qids01": {
        "level": None,
        "week": None,
        "days_baseline": [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "version_form": ["Self Rating", "Clinician"],
        "whitelist": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc', 'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf',
                      'vsuic', 'vintr', 'vengy', 'vslow', 'vagit', 'qstot'],
        "col_extenders": ['_week0_Self Rating', '_week2_Self Rating', '_week0_Clinician',
                          '_week2_Clinician'],
        "validation_params": {
            "num_cols_start": 17,
            "num_cols_end": 68
        },
    },
}

COL_NAME_SUBJECTKEY = "subjectkey"
COL_NAME_WEEK = "week"
COL_NAME_LEVEL = "level"
COL_NAME_DAYS_BASELINE = "days_baseline"
COL_NAME_VERSION_FORM = "version_form"

def get_scales_with_no_row_restrictions():
    output = []
    for key, dict in SCALES.items():
        if dict[COL_NAME_LEVEL] is None and dict[COL_NAME_WEEK] is None and dict[COL_NAME_DAYS_BASELINE] is None:
            output.append(key)
    return output

def get_whitelist():
    whitelist = []
    for key, dict in SCALES.items():
        if "col_extenders" in dict:
            for col_name in dict["whitelist"]:
                for extender in dict["col_extenders"]:
                    whitelist += [col_name + extender]
        else:
            whitelist += dict["whitelist"]
    return whitelist

# Define here so it gets instantiated at run-time
SCALES_NO_ROW_RESTRICTIONS = get_scales_with_no_row_restrictions()
WHITELIST = get_whitelist()

"""
Notes: Values that get converted to np.nan are being eliminated completely. Values that are converted to the ignr 
variable string still need to be determined.
"""
VALUE_CONVERSION_MAP = {
    "demo_-7": {
        "col_names": ['medicaid', 'privins', 'mkedc', 'enjoy', 'famim', 'volun', 'leave'],
        "values": {-7: ignr}
    },
    "publica": {
        "col_names": ['publica'],
        "values": {-7: ignr, 3: ignr}
    },
    "empl": {
        "col_names": ['empl'],
        "values": {15: ignr, 9: ignr, -7: ignr}
    },
    "student": {
        "col_names": ['student'],
        "values": {2: 0.5}
    },
    "educat": {
        "col_names": ['student'],
        "values": {999: ignr}
    },
    "thous": {
        "col_names": ['thous'],
        "values": {99: ignr}
    },
    "medication1_dosage": {
        "col_names": ['medication1_dosage'],
        "col_extenders": ['_Level 1_0.1', '_Level 1_2'],
        "values": {0: ignr, 999: ignr}
    },
    "crs01": {
        "col_names": ['heart', 'vsclr', 'hema', 'eyes', 'ugi', 'lgi', 'renal', 'genur', 'mskl', 'neuro', 'psych',
                      'respiratory', 'liverd', 'endod', 'hsoin', 'hmnin', 'hemin', 'hmdsd', 'hpanx', 'hinsg', 'happt',
                      'hwl', 'hsanx', 'hhypc', 'hvwsf', 'hsuic', 'hintr', 'hengy', 'hslow', 'hagit', 'hsex', 'suic_phx',
                      'drug_phx', 'alcohol', 'bip', 'dep', 'dage'],
        "values": {-9: ignr}
    },
    "blank_to_zero": {
        "col_names": ['sex_prs', 'gdiar', 'gcnst', 'gdmth', 'gnone', 'gnsea', 'gstro', 'htplp', 'htdzy', 'htchs', 'htnone',
                      'heart_prs', 'skrsh', 'skpsp', 'skich', 'sknone', 'skdry', 'nvhed', 'nvtrm', 'nvcrd', 'nvnone',
                      'nvdzy', 'nrvsy', 'eyvsn', 'earng', 'enone', 'eyear', 'urdif', 'urpn', 'urmns', 'urfrq', 'urnone',
                      'genur_prs', 'sldif', 'slnone', 'slmch', 'sleep', 'sxls', 'sxorg', 'sxerc', 'sxnone', 'oaxty',
                      'octrt', 'omal', 'orsls', 'oftge', 'odegy', 'onone', 'other_prs', 'skin_c', 'deppar', 'depsib',
                      'depchld', 'bippar', 'bipsib', 'bipchld', 'alcpar', 'alcsib', 'alcchld', 'drgpar', 'drgsib',
                      'drgchld', 'suicpar', 'suicsib', 'suicchld', 'fisfq', 'fisin', 'grseb', 'wpai02', 'wpai03',
                      'wpai04', 'wpai05', 'wpai_totalhrs', 'wpai_pctmissed', 'wpai_pctworked', 'wpai_pctwrkimp',
                      'wpai_pctactimp', 'wpai_totwrkimp', 'ucq010', 'ucq020', 'ucq030', 'ucq080', 'ucq091', 'ucq092',
                      'ucq100', 'ucq110', 'ucq120', 'ucq130', 'ucq140', 'ucq150', 'ucq160', 'ucq170', 'ucq040',
                      'ucq050', 'ucq060', 'ucq070'],
        "values": {"": 0}
    },
    "bulimia": {
        "col_names": ['bulimia'],
        "values": {0: np.nan, 1: np.nan, 2: "2/5", 5: "2/5"}
    },
    "zero_to_nan": {
        "col_names": ['ax_cocaine', 'alcoh', 'amphet', 'cannibis' , 'opioid'],
        "values": {0: np.nan}
    },
    "two_to_zero": {
        "col_names": ['wpai01', 'sfhs04', 'sfhs05', 'sfhs06', 'sfhs07', 'ucq010', 'ucq020', 'ucq080', 'ucq110',
                      'ucq120', 'ucq140', 'ucq160', 'ucq040', 'ucq060'],
        "values": {2: 0}
    },
    "sex_prs": {
        "col_names": ['sex_prs'],
        "values": {-7: 0, "": 0}
    },
    "qids01": {
        "col_names": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc', 'vapin', 'vwtdc', 'vwtin', 'vcntr',
                      'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit'],
        "col_extenders": ['_week0_Self Rating', '_week2_Self Rating', '_week0_Clinician',
                          '_week2_Clinician'],
        "values": {999: 0}
    },
    "minus": {
        6: ['sfhs12', 'sfhs11', 'sfhs10', 'sfhs09', 'sfhs01'], # Subtract 6 minus value
        3: ['sfhs02', 'sfhs03'], # Subtract 3 minus value
        1: ['sfhs08'] # Subtract 1
    },
}

COL_NAMES_ONE_HOT_ENCODE = set(['trtmt_Level 1_0.1', 'trtmt_Level 1_2', 'gender', 'resid', 'rtown', 'marital', 'bulimia',
                                'ax_cocaine', 'alcoh', 'amphet', 'cannibis' , 'opioid', 'empl', 'volun', 'leave',
                                'publica', 'medicaid', 'privins'])

