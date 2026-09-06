from datetime import datetime
import pandas as pd
import os
import json
from sklearn.ensemble import GradientBoostingClassifier

def print_with_tms(message):
    mytimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{mytimestamp}|{message}")

def read_data(path=".\\", file_name="data.csv", dtype_dict=None):
    mydata = pd.read_csv(os.path.join(path, file_name), dtype=dtype_dict)
    return mydata

def lending_club_dtype_dict():
    string_columns = [
        "term", "grade", "sub_grade", "emp_title", "emp_length",
        "home_ownership", "is_inc_v", "issue_d", "loan_status",
        "pymnt_plan", "url", "desc", "purpose", "title", "zip_code",
        "addr_state", "earliest_cr_line", "initial_list_status",
        "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
        "final_d", "status"
    ]

    integer_columns = [
        "id", "member_id", "loan_amnt", "funded_amnt", "policy_code",
        "not_compliant", "inactive_loans", "bad_loans", "emp_length_num",
        "delinq_2yrs_zero", "pub_rec_zero", "collections_12_mths_zero",
        "short_emp", "last_delinq_none", "last_record_none",
        "last_major_derog_none"
    ]

    float_columns = [
        "funded_amnt_inv", "int_rate", "installment", "annual_inc",
        "dti", "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
        "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
        "revol_util", "total_acc", "out_prncp", "out_prncp_inv",
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "last_pymnt_amnt",
        "collections_12_mths_ex_med", "mths_since_last_major_derog",
        "grade_num", "sub_grade_num", "payment_inc_ratio"
    ]

    return {
        **{column: "string" for column in string_columns},
        **{column: "Int64" for column in integer_columns},
        **{column: "float64" for column in float_columns},
    }

def assess_dataframe(df):
    print_with_tms(f"rowcount {df.shape[0]} colcount {df.shape[1]}")
    print_with_tms(f"dtypes {dict(df.dtypes)}")
    print_with_tms(f"columns\n{df.columns.tolist()}")
    if PRINTSTUFF == True:
        print_with_tms("First 5 rows:")
        print_with_tms(df.head())

def define_x_features():
    features = ['grade',                     # grade of the loan (categorical)
                'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
                'short_emp',                 # one year or less of employment
                'emp_length_num',            # number of years of employment
                'home_ownership',            # home_ownership status: own, mortgage or rent
                'dti',                       # debt to income ratio
                'purpose',                   # the purpose of the loan
                'payment_inc_ratio',         # ratio of the monthly payment to income
                'delinq_2yrs',               # number of delinquincies
                'delinq_2yrs_zero',          # no delinquincies in last 2 years
                'inq_last_6mths',            # number of creditor inquiries in last 6 months
                'last_delinq_none',          # has borrower had a delinquincy
                'last_major_derog_none',     # has borrower had 90 day or worse rating
                'open_acc',                  # number of open credit accounts
                'pub_rec',                   # number of derogatory public records
                'pub_rec_zero',              # no derogatory public records
                'revol_util',                # percent of available credit being used
                'total_rec_late_fee',        # total late fees received to day
                'int_rate',                  # interest rate of the loan
                'total_rec_int',             # interest received to date
                'annual_inc',                # annual income of borrower
                'funded_amnt',               # amount committed to the loan
                'funded_amnt_inv',           # amount committed by investors for the loan
                'installment',               # monthly payment owed by the borrower
            ]
    return features

def prepare_data(df, x_features, y_features):
    df2 = df.copy()
    df2['safe_loans'] = df2['bad_loans'].apply(lambda x : +1 if x==0 else -1)
    df2 = df2.drop(columns=["bad_loans"])
    return df2[x_features + y_features]

def apply_one_hot_encoding(df):
    '''
    The result of one-ho encoding is that multiclass categorical features are replaced with binary features
    that contain only 0 or 1 as values
    '''
    df2 = pd.get_dummies(df, dtype=int)
    return df2

def create_np_matrix(df, columnnames):
    '''
    Convert df, columns indicated by parameter columnnames, to numpy array (2D matrix)
    If there are N data points and D features then return a NxD matrix
    '''
    feature_matrix = df[columnnames].values
    print_with_tms(f"Created feature matrix with shape {feature_matrix.shape}")
    return feature_matrix

PRINTSTUFF = False
print_with_tms("Start script")
path = os.path.join("C:\\", "Users", "Evert Jan", "courseradatascience",
                       "course03", "module08", "data")

x_features = define_x_features()
y_features = ['safe_loans']

file_name_inp = "lending-club-data.csv"
all_data_df = read_data(path=path, file_name=file_name_inp, dtype_dict=lending_club_dtype_dict())
all_data_df = prepare_data(df=all_data_df, x_features=x_features, y_features=y_features)
all_data_df = all_data_df.dropna() # here only deletes 29 data points - 122578 remain

all_data_df = apply_one_hot_encoding(df=all_data_df)
x_features_enc = [cn for cn in all_data_df.columns.to_list() if cn not in y_features]

assess_dataframe(all_data_df)

#Use index files for train and val split (to reproduce a particular split)
#Important decision point: this happens AFTER dropna
with open(os.path.join(path, "module-8-assignment-1-train-idx.json")) as f:
    train_idx = json.load(f)
train_data_df = all_data_df.iloc[train_idx]
with open(os.path.join(path, "module-8-assignment-1-validation-idx.json")) as f:
    val_idx = json.load(f)
val_data_df = all_data_df.iloc[val_idx]
print_with_tms(f"total {len(all_data_df)} train {len(train_data_df)} val {len(val_data_df)}")

#Muller Guido page 90 - 94

X_train = create_np_matrix(df=train_data_df, columnnames=x_features_enc)
Y_train = create_np_matrix(df=train_data_df, columnnames=y_features)
X_val = create_np_matrix(df=val_data_df, columnnames=x_features_enc)
Y_val = create_np_matrix(df=val_data_df, columnnames=y_features)

model_5 = GradientBoostingClassifier(max_depth=6,
                                     n_estimators=5,
                                    )
model_5.fit(X=X_train,y=Y_train)

