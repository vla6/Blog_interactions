###########################################################
##
## Modify paths here to point to your data sources, 
## and locations for temporary or output files
##
## A few functions used for plotting are defined here also.
##
############################################################

# Input data
# Download from https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download
# and use the "accepted" file.  The path below should point to this file on your system.

input_path = './kaggle_input/accepted_2007_to_2018Q4.csv/accepted_2007_to_2018Q4.csv'

# Directory for temporary or intermediate files
temp_path = './data/2023_01_25'

###########################################################
##
## Constants.  These do not require modification.
## These are values used across notebooks, or are
## long and placed here for convenience
##
###########################################################

#
# Features to include in models
# Use a limited list to reduce runtime
# 

predictor_features = ['loan_amnt', 'term', 'int_rate', 'emp_length', 'home_ownership', 
                      'annual_inc', 'verification_status',
                      'fico_range_low', 'sec_app_fico_range_low', 
                      'open_acc', 'initial_list_status', 'num_actv_bc_tl',
                      'mort_acc', 'pub_rec', 'revol_bal']

#
# Features to retain in the dataset for informational purposes,
# beyond those used for predictions and the target
#

info_features = ['id', 'grade', 'sub_grade', 'hardship_flag', 'debt_settlement_flag',
                'hardship_amount', 'settlement_amount', 'addr_state', 'purpose'
                'total_rec_prncp', 'total_rec_int', 'tot_coll_amt', 'tot_cur_bal',
                'application_type']

#
# Some basic plotting functions - define here 
#

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

def plot_defaults():
    """ Set default plot parameters"""
    plt.style.use('seaborn-v0_8-white')
    
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams.update({'axes.titlesize': 16})

def plot_basic_bar(data, y, 
                   label = None,
                   n_bars = 10,
                   figsize = None,
                   ylabel = None,
                   title=None,
                   do_sort = False):
    """ Create a basic bar plot for a Pandas dataframe."""
    
    if do_sort:
        data = data.copy().sort_values(y, ascending=False)
        
    if label != None:
        data = data.copy().set_index(label)
        
    if ylabel == None:
        ylabel = data[y].name
        
    # Set figsize if not explicit
    if figsize == None:
        figsize = (4, n_bars/3.3)
        
    fig, ax = plt.subplots()
    
    data.head(n_bars)[[y]] \
        .plot(kind='barh', legend=None, figsize=figsize, ax=ax)
    
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel(None)
    ax.set_xlabel(ylabel)
    
    return fig

def plot_default_scale(x_data, thresh=3):
    """Simple function which decides whether to plot
    data using a linear or log scale.  If the skewness
    is past a threshold, use a log scale"""
    if (scipy.stats.skew(x_data) > thresh):
        return 'log'
    return 'linear'


#
# Input data dtypes for 01_data_import
#

input_dtypes = {'id' : 'str',
    'member_id' : 'str',
    'loan_amnt' : 'float',
    'funded_amnt' : 'float',
    'funded_amnt_inv' : 'str',
    'term' : 'str',
    'int_rate' : 'float',
    'installment' : 'float',
    'grade' : 'str',
    'sub_grade' : 'str',
    'emp_title' : 'str',
    'emp_length' : 'str',
    'home_ownership' : 'str',
    'annual_inc' : 'float',
    'verification_status' : 'str',
    'issue_d' : 'str',
    'loan_status' : 'str',
    'pymnt_plan' : 'str',
    'url' : 'str',
    'desc' : 'str',
    'purpose' : 'str',
    'title' : 'str',
    'zip_code' : 'str',
    'addr_state' : 'str',
    'dti' : 'float',
    'delinq_2yrs' : 'float',
    'earliest_cr_line' : 'str',
    'fico_range_low' : 'float',
    'fico_range_high' : 'float',
    'inq_last_6mths' : 'float',
    'mths_since_last_delinq' : 'float',
    'mths_since_last_record' : 'float',
    'open_acc' : 'float',
    'pub_rec' : 'float',
    'revol_bal' : 'float',
    'revol_util' : 'float',
    'total_acc' : 'float',
    'initial_list_status' : 'str',
    'out_prncp' : 'float',
    'out_prncp_inv' : 'float',
    'total_pymnt' : 'float',
    'total_pymnt_inv' : 'float',
    'total_rec_prncp' : 'float',
    'total_rec_int' : 'float',
    'total_rec_late_fee' : 'float',
    'recoveries' : 'float',
    'collection_recovery_fee' : 'float',
    'last_pymnt_d' : 'str',
    'last_pymnt_amnt' : 'float',
    'next_pymnt_d' : 'str',
    'last_credit_pull_d' : 'str',
    'last_fico_range_high' : 'float',
    'last_fico_range_low' : 'float',
    'collections_12_mths_ex_med' : 'float',
    'mths_since_last_major_derog' : 'float',
    'policy_code' : 'str',
    'application_type' : 'str',
    'annual_inc_joint' : 'float',
    'dti_joint' : 'float',
    'verification_status_joint' : 'str',
    'acc_now_delinq' : 'float',
    'tot_coll_amt' : 'float',
    'tot_cur_bal' : 'float',
    'open_acc_6m' : 'float',
    'open_act_il' : 'float',
    'open_il_12m' : 'float',
    'open_il_24m' : 'float',
    'mths_since_rcnt_il' : 'float',
    'total_bal_il' : 'float',
    'il_util' : 'float',
    'open_rv_12m' : 'float',
    'open_rv_24m' : 'float',
    'max_bal_bc' : 'float',
    'all_util' : 'float',
    'total_rev_hi_lim' : 'float',
    'inq_fi' : 'float',
    'total_cu_tl' : 'float',
    'inq_last_12m' : 'float',
    'acc_open_past_24mths' : 'float',
    'avg_cur_bal' : 'float',
    'bc_open_to_buy' : 'float',
    'bc_util' : 'float',
    'chargeoff_within_12_mths' : 'float',
    'delinq_amnt' : 'float',
    'mo_sin_old_il_acct' : 'float',
    'mo_sin_old_rev_tl_op' : 'float',
    'mo_sin_rcnt_rev_tl_op' : 'float',
    'mo_sin_rcnt_tl' : 'float',
    'mort_acc' : 'float',
    'mths_since_recent_bc' : 'float',
    'mths_since_recent_bc_dlq' : 'float',
    'mths_since_recent_inq' : 'float',
    'mths_since_recent_revol_delinq' : 'float',
    'num_accts_ever_120_pd' : 'float',
    'num_actv_bc_tl' : 'float',
    'num_actv_rev_tl' : 'float',
    'num_bc_sats' : 'float',
    'num_bc_tl' : 'float',
    'num_il_tl' : 'float',
    'num_op_rev_tl' : 'float',
    'num_rev_accts' : 'float',
    'num_rev_tl_bal_gt_0' : 'float',
    'num_sats' : 'float',
    'num_tl_120dpd_2m' : 'float',
    'num_tl_30dpd' : 'float',
    'num_tl_90g_dpd_24m' : 'float',
    'num_tl_op_past_12m' : 'float',
    'pct_tl_nvr_dlq' : 'float',
    'percent_bc_gt_75' : 'float',
    'pub_rec_bankruptcies' : 'float',
    'tax_liens' : 'float',
    'tot_hi_cred_lim' : 'float',
    'total_bal_ex_mort' : 'float',
    'total_bc_limit' : 'float',
    'total_il_high_credit_limit' : 'float',
    'revol_bal_joint' : 'float',
    'sec_app_fico_range_low' : 'float',
    'sec_app_fico_range_high' : 'float',
    'sec_app_earliest_cr_line' : 'str',
    'sec_app_inq_last_6mths' : 'float',
    'sec_app_mort_acc' : 'float',
    'sec_app_open_acc' : 'float',
    'sec_app_revol_util' : 'float',
    'sec_app_open_act_il' : 'float',
    'sec_app_num_rev_accts' : 'float',
    'sec_app_chargeoff_within_12_mths' : 'float',
    'sec_app_collections_12_mths_ex_med' : 'float',
    'sec_app_mths_since_last_major_derog' : 'float',
    'hardship_flag' : 'str',
    'hardship_type' : 'str',
    'hardship_reason' : 'str',
    'hardship_status' : 'str',
    'deferral_term' : 'float',
    'hardship_amount' : 'float',
    'hardship_start_date' : 'str',
    'hardship_end_date' : 'str',
    'payment_plan_start_date' : 'str',
    'hardship_length' : 'float',
    'hardship_dpd' : 'float',
    'hardship_loan_status' : 'str',
    'orig_projected_additional_accrued_interest' : 'str',
    'hardship_payoff_balance_amount' : 'float',
    'hardship_last_payment_amount' : 'float',
    'disbursement_method' : 'str',
    'debt_settlement_flag' : 'str',
    'debt_settlement_flag_date' : 'str',
    'settlement_status' : 'str',
    'settlement_date' : 'str',
    'settlement_amount' : 'float',
    'settlement_percentage' : 'float',
    'settlement_term' : 'float'}

input_dates = ['issue_d', 'earliest_cr_line', 'last_pymnt_d',
              'next_pymnt_d', 'last_credit_pull_d', 'sec_app_earliest_cr_line',
              'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
               'debt_settlement_flag_date', 'settlement_date']