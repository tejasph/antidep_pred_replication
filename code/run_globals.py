# -*- coding: utf-8 -*-
"""
Globals for data directory and results directory. Replace with your own!

Created on Sun Oct 18 11:44:04 2020

@author: jjnun
"""

# Directories for data and results
DATA_DIR = r"C:\Users\Tejas\Documents\star_project\processed_STARD_datasets"
RESULTS_DIR = r'C:\Users\Tejas\Documents\star_project\script_results'

#########################################
#Directories for regression related paths
#########################################

# Path for reading in processed data
REG_PROCESSED_DATA = r"C:\Users\Tejas\Documents\star_project\antidep_pred_replication\data"

# Output path for optimized models
OPTIMIZED_MODELS = r"C:\Users\Tejas\Documents\star_project\antidep_pred_replication\results\optimized_for_RMSE"

# Path for modelling data (ie. X_train, y_train...etc)
REG_MODEL_DATA_DIR = r"C:\Users\Tejas\Documents\star_project\antidep_pred_replication\data\modelling"

# Output path for experiment results
REG_RESULTS_DIR = r"C:\Users\Tejas\Documents\star_project\antidep_pred_replication\results\experiments"

# Variable Categories
cont_vars = ['interview_age','dm01_enroll__resm','dm01_enroll__relat','dm01_enroll__frend','dm01_enroll__thous',
             'dm01_enroll__educat','dm01_w0__mempl','dm01_w0__massist','dm01_w0__munempl','dm01_w0__minc_other',
             'dm01_w0__totincom','hrsd01__hdtot_r','phx01__dage','phx01__epino','phx01__episode_date','qids01_w0c__qstot',
             'qids01_w0sr__qstot','qids01_w2c__qstot','qids01_w2sr__qstot','qlesq01__totqlesq','sfhs01__pcs12',
             'sfhs01__mcs12','ucq01__ucq030','ucq01__ucq091','ucq01__ucq100','ucq01__ucq130','ucq01__ucq150','ucq01__ucq170',
             'ucq01__ucq050','ucq01__ucq070','wpai01__wpai02','wpai01__wpai03','wpai01__wpai04','wpai01__wpai_totalhrs',
             'wpai01__wpai_pctmissed','wpai01__wpai_pctworked','wpai01__wpai_pctwrkimp','wpai01__wpai_pctactimp',
             'wpai01__wpai_totwrkimp','wsas01__totwsas','imput_bech','imput_maier','imput_santen','imput_gibbons',
             'imput_hamd7','imput_hamdret','imput_hamdanx','imput_idsc5pccg','imput_qidscpccg']

categorical_vars = ['gender||F', 'gender||M', 'ccv01_w2__trtmt||1.0', 'ccv01_w2__trtmt||2.0', 'ccv01_w2__trtmt||3.0', 
'ccv01_w2__trtmt||4.0', 'dm01_enroll__resid||1.0', 'dm01_enroll__resid||2.0', 'dm01_enroll__resid||3.0',
 'dm01_enroll__resid||4.0', 'dm01_enroll__resid||5.0', 'dm01_enroll__resid||6.0', 'dm01_enroll__resid||7.0',
  'dm01_enroll__resid||8.0', 'dm01_enroll__rtown||1.0', 'dm01_enroll__rtown||2.0', 'dm01_enroll__rtown||3.0',
   'dm01_enroll__marital||1.0', 'dm01_enroll__marital||2.0', 'dm01_enroll__marital||3.0', 'dm01_enroll__marital||4.0',
    'dm01_enroll__marital||5.0', 'dm01_enroll__marital||6.0', 'dm01_enroll__empl||1.0', 'dm01_enroll__empl||2.0',
     'dm01_enroll__empl||3.0', 'dm01_enroll__empl||4.0', 'dm01_enroll__empl||5.0', 'dm01_enroll__empl||6.0', 
     'dm01_enroll__volun||0.0', 'dm01_enroll__volun||1.0', 'dm01_enroll__volun||2.0', 'dm01_enroll__leave||0.0',
      'dm01_enroll__leave||1.0', 'dm01_enroll__publica||0.0', 'dm01_enroll__publica||1.0', 'dm01_enroll__medicaid||0.0',
       'dm01_enroll__medicaid||1.0', 'dm01_enroll__privins||0.0', 'dm01_enroll__privins||1.0', 'idsc01__iwrse||1.0',
        'idsc01__iwrse||2.0', 'idsc01__iwrse||3.0', 'phx01__bulimia||2/5', 'phx01__bulimia||3', 'phx01__bulimia||4', 
        'phx01__alcoh||1.0', 'phx01__alcoh||2.0', 'phx01__amphet||1.0', 'phx01__amphet||2.0', 'phx01__cannibis||1.0',
         'phx01__cannibis||2.0', 'phx01__opioid||1.0', 'phx01__opioid||2.0', 'phx01__ax_cocaine||1.0', 'phx01__ax_cocaine||2.0']

categorical_dict ={'gender':2, 'ccv01_w2__trtmt':4, 'dm01_enroll__resid':8,'dm01_enroll__rtown':3,'dm01_enroll__marital':6,
           'dm01_enroll__empl':6,'dm01_enroll__volun':3,'dm01_enroll__leave':2,'dm01_enroll__publica':2,'dm01_enroll__medicaid':2,
           'dm01_enroll__privins':2,'idsc01__iwrse':3,'phx01__bulimia':3,'phx01__alcoh':2,'phx01__amphet':2,'phx01__cannibis':2,'phx01__opioid':2,
           'phx01__ax_cocaine':2}

ord_vars = ['crs01__heart','crs01__vsclr','crs01__hema','crs01__eyes','crs01__ugi','crs01__lgi','crs01__renal',
           'crs01__genur','crs01__mskl','crs01__neuro','crs01__psych','crs01__respiratory','crs01__liverd',
            'crs01__endod','ccv01_w2__medication1_dosage','dm01_enroll__student','dm01_enroll__mkedc','dm01_enroll__enjoy',
            'dm01_enroll__famim','hrsd01__hsoin','hrsd01__hmnin','hrsd01__hemin','hrsd01__hmdsd','hrsd01__hpanx','hrsd01__hinsg',
            'hrsd01__happt','hrsd01__hwl','hrsd01__hsanx','hrsd01__hhypc','hrsd01__hvwsf','hrsd01__hsuic','hrsd01__hintr',
            'hrsd01__hengy','hrsd01__hslow','hrsd01__hagit','hrsd01__hsex','idsc01__isoin','idsc01__imnin','idsc01__iemin',
            'idsc01__ihysm','idsc01__imdsd','idsc01__ianx','idsc01__ipanc','idsc01__iirtb','idsc01__irct','idsc01__ivrtn',
           'idsc01__iqty','idsc01__iapdc','idsc01__iapin','idsc01__iwtdc','idsc01__iwtin','idsc01__icntr','idsc01__ivwsf',
           'idsc01__ivwfr','idsc01__isuic','idsc01__iintr','idsc01__iplsr','idsc01__iengy','idsc01__isex','idsc01__islow',
           'idsc01__iagit','idsc01__ismtc','idsc01__isymp','idsc01__igas','idsc01__iintp','idsc01__ildn','qids01_w0c__vsoin',
            'qids01_w0c__vmnin','qids01_w0c__vemin','qids01_w0c__vhysm','qids01_w0c__vmdsd','qids01_w0c__vapdc','qids01_w0c__vapin',
           'qids01_w0c__vwtdc','qids01_w0c__vwtin','qids01_w0c__vcntr','qids01_w0c__vvwsf','qids01_w0c__vsuic','qids01_w0c__vintr',
           'qids01_w0c__vengy','qids01_w0c__vslow','qids01_w0c__vagit','qids01_w0sr__vsoin','qids01_w0sr__vmnin','qids01_w0sr__vemin',
            'qids01_w0sr__vhysm','qids01_w0sr__vmdsd','qids01_w0sr__vapdc','qids01_w0sr__vapin','qids01_w0sr__vwtdc','qids01_w0sr__vwtin',
            'qids01_w0sr__vcntr','qids01_w0sr__vvwsf','qids01_w0sr__vsuic','qids01_w0sr__vintr','qids01_w0sr__vengy','qids01_w0sr__vslow',
           'qids01_w0sr__vagit','qids01_w2c__vsoin','qids01_w2c__vmnin','qids01_w2c__vemin','qids01_w2c__vhysm','qids01_w2c__vmdsd',
           'qids01_w2c__vapdc','qids01_w2c__vapin','qids01_w2c__vwtdc','qids01_w2c__vwtin','qids01_w2c__vcntr','qids01_w2c__vvwsf',
            'qids01_w2c__vsuic','qids01_w2c__vintr','qids01_w2c__vengy','qids01_w2c__vslow','qids01_w2c__vagit','qids01_w2sr__vsoin',
            'qids01_w2sr__vmnin','qids01_w2sr__vemin','qids01_w2sr__vhysm','qids01_w2sr__vmdsd','qids01_w2sr__vapdc','qids01_w2sr__vapin',
           'qids01_w2sr__vwtdc','qids01_w2sr__vwtin','qids01_w2sr__vcntr','qids01_w2sr__vvwsf','qids01_w2sr__vsuic','qids01_w2sr__vintr',
            'qids01_w2sr__vengy','qids01_w2sr__vslow','qids01_w2sr__vagit','qlesq01__qlesq01','qlesq01__qlesq02','qlesq01__qlesq03',
           'qlesq01__qlesq04','qlesq01__qlesq05','qlesq01__qlesq06','qlesq01__qlesq07','qlesq01__qlesq08','qlesq01__qlesq09','qlesq01__qlesq10',
           'qlesq01__qlesq11','qlesq01__qlesq12','qlesq01__qlesq13','qlesq01__qlesq14','qlesq01__qlesq15','qlesq01__qlesq16',
           'sfhs01__sfhs01','sfhs01__sfhs02','sfhs01__sfhs03','sfhs01__sfhs08','sfhs01__sfhs09','sfhs01__sfhs10','sfhs01__sfhs11',
           'sfhs01__sfhs12','side_effects01__fisfq','side_effects01__fisin','side_effects01__grseb','ucq01__ucq092','wpai01__wpai05',
            'wpai01__wpai06','wsas01__wsas01','wsas01__wsas02','wsas01__wsas03','wsas01__wsas04','wsas01__wsas05','imput_hamdsle','imput_idsc5w0','imput_idsc5w2']


binary_vars =['ccv01_w2__suicd', 'ccv01_w2__remsn', 'ccv01_w2__raise', 'ccv01_w2__effct', 'ccv01_w2__cncn', 'ccv01_w2__prtcl', 'ccv01_w2__stmed', 'dm01_enroll__spous',
 'dm01_w0__inc_curr', 'dm01_w0__assist', 'dm01_w0__unempl', 'dm01_w0__otherinc', 'idsc01__ienv', 'mhx01__psmed', 'pdsq01__evy2w', 'pdsq01__joy2w', 'pdsq01__int2w',
  'pdsq01__lap2w', 'pdsq01__gap2w', 'pdsq01__lsl2w', 'pdsq01__msl2w', 'pdsq01__jmp2w', 'pdsq01__trd2w', 'pdsq01__glt2w', 'pdsq01__neg2w', 'pdsq01__flr2w', 'pdsq01__cnt2w',
   'pdsq01__dcn2w', 'pdsq01__psv2w', 'pdsq01__wsh2w', 'pdsq01__btr2w', 'pdsq01__tht2w', 'pdsq01__ser2w', 'pdsq01__spf2w', 'pdsq01__sad2y', 'pdsq01__apt2y', 'pdsq01__slp2y', 
   'pdsq01__trd2y', 'pdsq01__cd2y', 'pdsq01__low2y', 'pdsq01__hpl2y', 'pdsq01__trexp', 'pdsq01__trwit', 'pdsq01__tetht', 'pdsq01__teups', 'pdsq01__temem', 'pdsq01__tedis',
    'pdsq01__teblk', 'pdsq01__termd', 'pdsq01__tefsh', 'pdsq01__teshk', 'pdsq01__tedst', 'pdsq01__tenmb', 'pdsq01__tegug', 'pdsq01__tegrd', 'pdsq01__tejmp', 'pdsq01__ebnge',
     'pdsq01__ebcrl', 'pdsq01__ebfl', 'pdsq01__ebhgy', 'pdsq01__ebaln', 'pdsq01__ebdsg', 'pdsq01__ebups', 'pdsq01__ebdt', 'pdsq01__ebvmt', 'pdsq01__ebwgh', 'pdsq01__obgrm', 
     'pdsq01__obfgt', 'pdsq01__obvlt', 'pdsq01__obstp', 'pdsq01__obint', 'pdsq01__obcln', 'pdsq01__obrpt', 'pdsq01__obcnt', 'pdsq01__anhrt', 'pdsq01__anbrt', 'pdsq01__anshk',
      'pdsq01__anrsn', 'pdsq01__anczy', 'pdsq01__ansym', 'pdsq01__anwor', 'pdsq01__anavd', 'pdsq01__pechr', 'pdsq01__pecnf', 'pdsq01__peslp', 'pdsq01__petlk', 'pdsq01__pevth',
       'pdsq01__peimp', 'pdsq01__imagn', 'pdsq01__imspy', 'pdsq01__imdgr', 'pdsq01__impwr', 'pdsq01__imcrl', 'pdsq01__imvcs', 'pdsq01__fravd', 'pdsq01__frfar', 'pdsq01__frcwd',
        'pdsq01__frlne', 'pdsq01__frbrg', 'pdsq01__frbus', 'pdsq01__frcar', 'pdsq01__fralo', 'pdsq01__fropn', 'pdsq01__franx', 'pdsq01__frsit', 'pdsq01__emwry', 'pdsq01__emstu',
         'pdsq01__ematn', 'pdsq01__emsoc', 'pdsq01__emavd', 'pdsq01__emspk', 'pdsq01__emeat', 'pdsq01__emupr', 'pdsq01__emwrt', 'pdsq01__emstp', 'pdsq01__emqst', 'pdsq01__embmt',
          'pdsq01__empty', 'pdsq01__emanx', 'pdsq01__emsit', 'pdsq01__dkmch', 'pdsq01__dkfam', 'pdsq01__dkfrd', 'pdsq01__dkcut', 'pdsq01__dkpbm', 'pdsq01__dkmge', 'pdsq01__dgmch',
           'pdsq01__dgfam', 'pdsq01__dgfrd', 'pdsq01__dgcut', 'pdsq01__dgpbm', 'pdsq01__dgmge', 'pdsq01__wynrv', 'pdsq01__wybad', 'pdsq01__wysdt', 'pdsq01__wydly', 'pdsq01__wyrst',
            'pdsq01__wyslp', 'pdsq01__wytsn', 'pdsq01__wycnt', 'pdsq01__wysnp', 'pdsq01__wycrl', 'pdsq01__phstm', 'pdsq01__phach', 'pdsq01__phsck', 'pdsq01__phpr', 'pdsq01__phcse',
             'pdsq01__wiser', 'pdsq01__wistp', 'pdsq01__wiill', 'pdsq01__wintr', 'pdsq01__widr', 'phx01__ai_none', 'phx01__pd_ag', 'phx01__pd_noag', 'phx01__specphob', 'phx01__soc_phob',
              'phx01__ocd_phx', 'phx01__psd', 'phx01__gad_phx', 'phx01__axi_oth', 'phx01__aii_none', 'phx01__aii_def', 'phx01__aii_na', 'phx01__pd_border', 'phx01__pd_depend',
               'phx01__pd_antis', 'phx01__pd_paran', 'phx01__pd_nos', 'phx01__axii_oth', 'phx01__dep', 'phx01__deppar', 'phx01__depsib', 'phx01__depchld', 'phx01__bip', 'phx01__bippar',
                'phx01__bipsib', 'phx01__bipchld', 'phx01__alcohol', 'phx01__alcpar', 'phx01__alcsib', 'phx01__alcchld', 'phx01__drug_phx', 'phx01__drgpar', 'phx01__drgsib',
                 'phx01__drgchld', 'phx01__suic_phx', 'phx01__suicpar', 'phx01__suicsib', 'phx01__suicchld', 'phx01__wrsms', 'phx01__anorexia', 'sfhs01__sfhs04', 'sfhs01__sfhs05',
                  'sfhs01__sfhs06', 'sfhs01__sfhs07', 'ucq01__ucq010', 'ucq01__ucq020', 'ucq01__ucq080', 'ucq01__ucq110', 'ucq01__ucq120', 'ucq01__ucq140', 'ucq01__ucq160', 
                  'ucq01__ucq040', 'ucq01__ucq060', 'wpai01__wpai01', 'imput_anyanxiety']

